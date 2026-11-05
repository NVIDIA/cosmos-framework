// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#include <torch/extension.h>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// Minimal parquet footer + page header scanner.
// Extracts only what the GPU pipeline needs: per-column page offsets,
// compressed/uncompressed sizes, page types, num_values, and rep/def levels.
// Works directly on the raw parquet bytes (pinned memory buffer).

namespace {

// Thrift compact protocol helpers
inline int64_t read_varint(const uint8_t* buf, int64_t& pos, int64_t end) {
    int64_t result = 0;
    int shift = 0;
    while (pos < end) {
        uint8_t b = buf[pos++];
        result |= (int64_t)(b & 0x7F) << shift;
        if ((b & 0x80) == 0) break;
        shift += 7;
    }
    return result;
}

inline int64_t zigzag_decode(int64_t n) {
    return (n >> 1) ^ -(n & 1);
}

// Skip a thrift compact field value
void skip_thrift_field(const uint8_t* buf, int64_t& pos, int64_t end, int type_id) {
    if (type_id <= 2) return; // bool
    if (type_id == 3) { pos++; return; } // i8
    if (type_id >= 4 && type_id <= 6) { read_varint(buf, pos, end); return; } // i16/i32/i64
    if (type_id == 7) { pos += 8; return; } // double
    if (type_id == 8) { // binary
        int64_t len = read_varint(buf, pos, end);
        pos += len;
        return;
    }
    if (type_id == 9 || type_id == 10) { // list/set
        uint8_t h = buf[pos++];
        int64_t size = (h >> 4) & 0x0F;
        int elem_type = h & 0x0F;
        if (size == 15) size = read_varint(buf, pos, end);
        for (int64_t i = 0; i < size; i++) skip_thrift_field(buf, pos, end, elem_type);
        return;
    }
    if (type_id == 11) { // map
        int64_t size = read_varint(buf, pos, end);
        if (size > 0) {
            uint8_t types = buf[pos++];
            int key_type = (types >> 4) & 0x0F;
            int val_type = types & 0x0F;
            for (int64_t i = 0; i < size; i++) {
                skip_thrift_field(buf, pos, end, key_type);
                skip_thrift_field(buf, pos, end, val_type);
            }
        }
        return;
    }
    if (type_id == 12) { // struct
        while (pos < end) {
            uint8_t b = buf[pos++];
            if (b == 0) break;
            int ft = b & 0x0F;
            int delta = (b >> 4) & 0x0F;
            if (delta == 0) read_varint(buf, pos, end);
            skip_thrift_field(buf, pos, end, ft);
        }
        return;
    }
}

// Parse a parquet PageHeader, extracting page_type, uncomp_size, comp_size
struct PageHeaderResult {
    int page_type;
    int uncompressed_size;
    int compressed_size;
    int header_bytes;
};

PageHeaderResult parse_page_header(const uint8_t* buf, int64_t offset, int64_t end) {
    PageHeaderResult r{-1, -1, -1, 0};
    int64_t pos = offset;
    int prev_fid = 0;

    while (pos < end) {
        uint8_t byte = buf[pos++];
        if (byte == 0) break;

        int type_id = byte & 0x0F;
        int delta = (byte >> 4) & 0x0F;
        int cur_fid;

        if (delta == 0) {
            cur_fid = (int)zigzag_decode(read_varint(buf, pos, end));
        } else {
            cur_fid = prev_fid + delta;
        }
        prev_fid = cur_fid;

        if (cur_fid == 1) { // PageType
            r.page_type = (int)zigzag_decode(read_varint(buf, pos, end));
        } else if (cur_fid == 2) { // uncompressed_page_size
            r.uncompressed_size = (int)zigzag_decode(read_varint(buf, pos, end));
        } else if (cur_fid == 3) { // compressed_page_size
            r.compressed_size = (int)zigzag_decode(read_varint(buf, pos, end));
        } else {
            skip_thrift_field(buf, pos, end, type_id);
        }
    }
    r.header_bytes = (int)(pos - offset);
    return r;
}

// Thrift compact struct reader for the FileMetaData footer
struct ColumnChunkMeta {
    std::string path;
    int64_t num_values = 0;
    int64_t dict_page_offset = -1;
    int64_t data_page_offset = 0;
    int64_t total_compressed_size = 0;
};

// Read a thrift string (binary field)
std::string read_thrift_string(const uint8_t* buf, int64_t& pos, int64_t end) {
    int64_t len = read_varint(buf, pos, end);
    std::string s((const char*)buf + pos, len);
    pos += len;
    return s;
}

// Parse ColumnMetaData (the inner struct of ColumnChunk)
ColumnChunkMeta parse_column_metadata(const uint8_t* buf, int64_t& pos, int64_t end) {
    ColumnChunkMeta cm;
    int prev_fid = 0;
    std::vector<std::string> path_parts;

    while (pos < end) {
        uint8_t byte = buf[pos++];
        if (byte == 0) break;

        int type_id = byte & 0x0F;
        int delta = (byte >> 4) & 0x0F;
        int cur_fid;
        if (delta == 0) {
            cur_fid = (int)zigzag_decode(read_varint(buf, pos, end));
        } else {
            cur_fid = prev_fid + delta;
        }
        prev_fid = cur_fid;

        switch (cur_fid) {
            case 1: // type (enum)
                read_varint(buf, pos, end);
                break;
            case 2: // encodings (list<enum>)
                skip_thrift_field(buf, pos, end, 9);
                break;
            case 3: // path_in_schema (list<string>)
            {
                uint8_t h = buf[pos++];
                int64_t size = (h >> 4) & 0x0F;
                if (size == 15) size = read_varint(buf, pos, end);
                for (int64_t i = 0; i < size; i++) {
                    path_parts.push_back(read_thrift_string(buf, pos, end));
                }
                break;
            }
            case 4: // codec (enum)
                read_varint(buf, pos, end);
                break;
            case 5: // num_values
                cm.num_values = zigzag_decode(read_varint(buf, pos, end));
                break;
            case 6: // total_uncompressed_size
                zigzag_decode(read_varint(buf, pos, end));
                break;
            case 7: // total_compressed_size
                cm.total_compressed_size = zigzag_decode(read_varint(buf, pos, end));
                break;
            case 8: // key_value_metadata
                skip_thrift_field(buf, pos, end, type_id);
                break;
            case 9: // data_page_offset
                cm.data_page_offset = zigzag_decode(read_varint(buf, pos, end));
                break;
            case 10: // index_page_offset
                zigzag_decode(read_varint(buf, pos, end));
                break;
            case 11: // dictionary_page_offset
                cm.dict_page_offset = zigzag_decode(read_varint(buf, pos, end));
                break;
            default:
                skip_thrift_field(buf, pos, end, type_id);
                break;
        }
    }

    // Build dotted path
    for (size_t i = 0; i < path_parts.size(); i++) {
        if (i > 0) cm.path += ".";
        cm.path += path_parts[i];
    }
    return cm;
}

// Schema element for extracting rep/def levels and physical types
struct SchemaElement {
    std::string name;
    int repetition_type = 0; // 0=REQUIRED, 1=OPTIONAL, 2=REPEATED
    int num_children = 0;
    int physical_type = -1;  // parquet type: 0=BOOL,1=INT32,2=INT64,3=INT96,4=FLOAT,5=DOUBLE,6=BYTE_ARRAY,7=FIXED
};

SchemaElement parse_schema_element(const uint8_t* buf, int64_t& pos, int64_t end) {
    SchemaElement se;
    int prev_fid = 0;
    while (pos < end) {
        uint8_t byte = buf[pos++];
        if (byte == 0) break;
        int type_id = byte & 0x0F;
        int delta = (byte >> 4) & 0x0F;
        int cur_fid;
        if (delta == 0) {
            cur_fid = (int)zigzag_decode(read_varint(buf, pos, end));
        } else {
            cur_fid = prev_fid + delta;
        }
        prev_fid = cur_fid;

        switch (cur_fid) {
            case 1: // type (enum) — only present for leaf columns
                se.physical_type = (int)zigzag_decode(read_varint(buf, pos, end));
                break;
            case 2: // type_length
                zigzag_decode(read_varint(buf, pos, end));
                break;
            case 3: // repetition_type (enum)
                se.repetition_type = (int)zigzag_decode(read_varint(buf, pos, end));
                break;
            case 4: // name
                se.name = read_thrift_string(buf, pos, end);
                break;
            case 5: // num_children
                se.num_children = (int)zigzag_decode(read_varint(buf, pos, end));
                break;
            default:
                skip_thrift_field(buf, pos, end, type_id);
                break;
        }
    }
    return se;
}

struct ColInfo {
    int max_rep, max_def;
    int physical_type; // -1 if not a leaf
};

// Compute max_rep, max_def, and physical_type from schema tree
void compute_rep_def(
    const std::vector<SchemaElement>& schema,
    int idx, int rep, int def,
    const std::string& path_prefix,
    std::unordered_map<std::string, ColInfo>& result
) {
    const auto& se = schema[idx];
    int new_rep = rep + (se.repetition_type == 2 ? 1 : 0);
    int new_def = def + (se.repetition_type != 0 ? 1 : 0);

    std::string path = path_prefix.empty() ? se.name : path_prefix + "." + se.name;

    if (se.num_children == 0) {
        result[path] = {new_rep, new_def, se.physical_type};
    } else {
        int child = idx + 1;
        for (int c = 0; c < se.num_children; c++) {
            compute_rep_def(schema, child, new_rep, new_def, path, result);
            std::function<int(int)> subtree_size = [&](int i) -> int {
                int count = 1;
                for (int j = 0; j < schema[i].num_children; j++) {
                    count += subtree_size(i + count);
                }
                return count;
            };
            child += subtree_size(child);
        }
    }
}

} // anonymous namespace


// scan_parquet_pages_cpp: scan one parquet file's pages from raw bytes.
//
// Input: pinned uint8 tensor, file offset, file size, list of column paths to scan
//
// Returns a flat tensor with per-column results:
//   For each column: [num_values, max_rep, max_def,
//                     dict_page_type, dict_data_offset, dict_comp_size, dict_uncomp_size,
//                     n_data_pages,
//                     data_page_0_type, data_page_0_data_offset, data_page_0_comp, data_page_0_uncomp,
//                     ...]
// Returns an empty tensor if the file can't be parsed.
torch::Tensor scan_parquet_pages_cpp(
    torch::Tensor pinned_buf,
    int64_t file_offset,
    int64_t file_size,
    std::vector<std::string> column_paths
) {
    const uint8_t* buf = pinned_buf.data_ptr<uint8_t>() + file_offset;
    int64_t end = file_size;

    // Read footer length (last 8 bytes: 4-byte footer length + "PAR1")
    if (end < 12) return torch::empty({0}, torch::kInt64);
    if (buf[end-1] != '1' || buf[end-2] != 'R' || buf[end-3] != 'A' || buf[end-4] != 'P')
        return torch::empty({0}, torch::kInt64);

    int32_t footer_len;
    std::memcpy(&footer_len, buf + end - 8, 4);
    if (footer_len <= 0 || footer_len > end - 8)
        return torch::empty({0}, torch::kInt64);

    int64_t footer_start = end - 8 - footer_len;
    int64_t fpos = footer_start;

    // Parse FileMetaData thrift struct
    int prev_fid = 0;
    std::vector<SchemaElement> schema;

    // Row group column chunks
    struct RowGroupInfo {
        std::vector<ColumnChunkMeta> columns;
    };
    std::vector<RowGroupInfo> row_groups;

    while (fpos < end) {
        uint8_t byte = buf[fpos++];
        if (byte == 0) break;

        int type_id = byte & 0x0F;
        int delta = (byte >> 4) & 0x0F;
        int cur_fid;
        if (delta == 0) {
            cur_fid = (int)zigzag_decode(read_varint(buf, fpos, end));
        } else {
            cur_fid = prev_fid + delta;
        }
        prev_fid = cur_fid;

        switch (cur_fid) {
            case 1: // version
                zigzag_decode(read_varint(buf, fpos, end));
                break;
            case 2: // schema (list<SchemaElement>)
            {
                uint8_t h = buf[fpos++];
                int64_t size = (h >> 4) & 0x0F;
                if (size == 15) size = read_varint(buf, fpos, end);
                for (int64_t i = 0; i < size; i++) {
                    schema.push_back(parse_schema_element(buf, fpos, end));
                }
                break;
            }
            case 3: // num_rows
                zigzag_decode(read_varint(buf, fpos, end));
                break;
            case 4: // row_groups (list<RowGroup>)
            {
                uint8_t h = buf[fpos++];
                int64_t n_rg = (h >> 4) & 0x0F;
                if (n_rg == 15) n_rg = read_varint(buf, fpos, end);

                for (int64_t rg = 0; rg < n_rg; rg++) {
                    RowGroupInfo rgi;
                    int rg_prev = 0;
                    while (fpos < end) {
                        uint8_t b2 = buf[fpos++];
                        if (b2 == 0) break;
                        int t2 = b2 & 0x0F;
                        int d2 = (b2 >> 4) & 0x0F;
                        int f2;
                        if (d2 == 0) {
                            f2 = (int)zigzag_decode(read_varint(buf, fpos, end));
                        } else {
                            f2 = rg_prev + d2;
                        }
                        rg_prev = f2;

                        if (f2 == 1) { // columns (list<ColumnChunk>)
                            uint8_t h2 = buf[fpos++];
                            int64_t n_col = (h2 >> 4) & 0x0F;
                            if (n_col == 15) n_col = read_varint(buf, fpos, end);

                            for (int64_t ci = 0; ci < n_col; ci++) {
                                int cc_prev = 0;
                                ColumnChunkMeta ccm;
                                while (fpos < end) {
                                    uint8_t b3 = buf[fpos++];
                                    if (b3 == 0) break;
                                    int t3 = b3 & 0x0F;
                                    int d3 = (b3 >> 4) & 0x0F;
                                    int f3;
                                    if (d3 == 0) {
                                        f3 = (int)zigzag_decode(read_varint(buf, fpos, end));
                                    } else {
                                        f3 = cc_prev + d3;
                                    }
                                    cc_prev = f3;

                                    if (f3 == 3) { // meta_data (ColumnMetaData struct)
                                        ccm = parse_column_metadata(buf, fpos, end);
                                    } else {
                                        skip_thrift_field(buf, fpos, end, t3);
                                    }
                                }
                                rgi.columns.push_back(ccm);
                            }
                        } else {
                            skip_thrift_field(buf, fpos, end, t2);
                        }
                    }
                    row_groups.push_back(rgi);
                }
                break;
            }
            default:
                skip_thrift_field(buf, fpos, end, type_id);
                break;
        }
    }

    // Compute rep/def levels + physical types from schema
    std::unordered_map<std::string, ColInfo> col_info_map;
    if (!schema.empty() && schema[0].num_children > 0) {
        int child = 1;
        for (int c = 0; c < schema[0].num_children; c++) {
            compute_rep_def(schema, child, 0, 0, "", col_info_map);
            std::function<int(int)> subtree_size = [&](int i) -> int {
                int count = 1;
                for (int j = 0; j < schema[i].num_children; j++) {
                    count += subtree_size(i + count);
                }
                return count;
            };
            child += subtree_size(child);
        }
    }

    // For each requested column, scan page headers
    // Output format per column: [num_values, max_rep, max_def, physical_type,
    //   has_dict, dict_data_offset, dict_comp, dict_uncomp,
    //   n_data_pages,
    //   (data_offset, comp_size, uncomp_size) x n_data_pages]
    std::vector<int64_t> output;

    for (const auto& col_path : column_paths) {
        int64_t total_num_values = 0;
        int max_rep = 0, max_def = 0;
        int physical_type = -1;
        bool has_dict = false;
        int64_t dict_data_offset = -1, dict_comp = 0, dict_uncomp = 0;
        struct DataPageInfo { int64_t data_offset, comp, uncomp; };
        std::vector<DataPageInfo> all_data_pages;
        bool found_any = false;

        auto it = col_info_map.find(col_path);
        if (it != col_info_map.end()) {
            max_rep = it->second.max_rep;
            max_def = it->second.max_def;
            physical_type = it->second.physical_type;
        }

        // Scan ALL row groups for this column
        for (const auto& rg : row_groups) {
            for (const auto& ccm : rg.columns) {
                if (ccm.path != col_path) continue;
                found_any = true;
                total_num_values += ccm.num_values;

                int64_t chunk_start = (ccm.dict_page_offset >= 0)
                    ? ccm.dict_page_offset : ccm.data_page_offset;
                int64_t chunk_end = chunk_start + ccm.total_compressed_size;

                int64_t scan_pos = chunk_start;
                while (scan_pos < chunk_end) {
                    auto phr = parse_page_header(buf, scan_pos, end);
                    if (phr.compressed_size < 0) break;

                    int64_t data_off = scan_pos + phr.header_bytes;
                    if (phr.page_type == 2) { // DICTIONARY_PAGE
                        if (!has_dict) {
                            has_dict = true;
                            dict_data_offset = data_off;
                            dict_comp = phr.compressed_size;
                            dict_uncomp = phr.uncompressed_size;
                        }
                    } else {
                        all_data_pages.push_back({
                            data_off,
                            phr.compressed_size,
                            phr.uncompressed_size
                        });
                    }
                    scan_pos = data_off + phr.compressed_size;
                }
                break; // found column in this row group, next rg
            }
        }

        if (found_any) {
            output.push_back(total_num_values);
            output.push_back(max_rep);
            output.push_back(max_def);
            output.push_back((int64_t)physical_type);
            output.push_back(has_dict ? 1 : 0);
            output.push_back(has_dict ? dict_data_offset : 0);
            output.push_back(dict_comp);
            output.push_back(dict_uncomp);
            output.push_back((int64_t)all_data_pages.size());
            for (const auto& dp : all_data_pages) {
                output.push_back(dp.data_offset);
                output.push_back(dp.comp);
                output.push_back(dp.uncomp);
            }
        } else {
            output.push_back(-1);
        }
    }

    auto result = torch::empty({(int64_t)output.size()}, torch::kInt64);
    std::memcpy(result.data_ptr<int64_t>(), output.data(), output.size() * sizeof(int64_t));
    return result;
}


#ifndef SCENE_LOADER_BUILD
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scan_parquet_pages_cpp", &scan_parquet_pages_cpp,
          "Scan parquet page metadata from raw bytes (GIL-free)",
          py::call_guard<py::gil_scoped_release>());
}
#endif
