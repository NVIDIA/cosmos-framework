// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

struct JsonValue {
    enum Type { J_NULL, J_BOOL, J_NUMBER, J_STRING, J_ARRAY, J_OBJECT };
    Type type = J_NULL;
    bool bool_val = false;
    double num_val = 0;
    std::string str_val;
    std::unique_ptr<std::vector<JsonValue>> arr_ptr;
    std::unique_ptr<std::unordered_map<std::string, JsonValue>> obj_ptr;

    JsonValue() = default;
    JsonValue(JsonValue&&) = default;
    JsonValue& operator=(JsonValue&&) = default;
    JsonValue(const JsonValue&) = delete;
    JsonValue& operator=(const JsonValue&) = delete;

    bool is_null()   const { return type == J_NULL; }
    bool is_object() const { return type == J_OBJECT; }
    bool is_array()  const { return type == J_ARRAY; }
    bool is_string() const { return type == J_STRING; }
    bool is_number() const { return type == J_NUMBER; }
    bool is_bool()   const { return type == J_BOOL; }

    double number() const { return num_val; }
    const std::string& str() const { return str_val; }

    size_t size() const {
        if (type == J_ARRAY && arr_ptr) return arr_ptr->size();
        if (type == J_OBJECT && obj_ptr) return obj_ptr->size();
        return 0;
    }

    bool has(const std::string& key) const {
        return type == J_OBJECT && obj_ptr && obj_ptr->count(key) > 0;
    }

    const JsonValue& operator[](const std::string& key) const {
        static const JsonValue null_v;
        if (!obj_ptr) return null_v;
        auto it = obj_ptr->find(key);
        return it != obj_ptr->end() ? it->second : null_v;
    }
    const JsonValue& operator[](size_t i) const {
        static const JsonValue null_v;
        if (!arr_ptr || i >= arr_ptr->size()) return null_v;
        return (*arr_ptr)[i];
    }

    double get_number(const std::string& key, double def = 0.0) const {
        if (!obj_ptr) return def;
        auto it = obj_ptr->find(key);
        if (it == obj_ptr->end()) return def;
        if (it->second.is_number()) return it->second.num_val;
        if (it->second.is_string()) return std::strtod(it->second.str_val.c_str(), nullptr);
        return def;
    }
    std::string get_string(const std::string& key, const std::string& def = "") const {
        if (!obj_ptr) return def;
        auto it = obj_ptr->find(key);
        if (it == obj_ptr->end() || !it->second.is_string()) return def;
        return it->second.str_val;
    }
};

namespace json_detail {

inline void skip_ws(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
}

inline std::string parse_string(const char*& p, const char* end) {
    if (p >= end || *p != '"') throw std::runtime_error("json: expected '\"'");
    ++p;
    std::string s;
    while (p < end && *p != '"') {
        if (*p == '\\') {
            ++p;
            if (p >= end) break;
            switch (*p) {
                case '"':  s += '"'; break;
                case '\\': s += '\\'; break;
                case '/':  s += '/'; break;
                case 'n':  s += '\n'; break;
                case 't':  s += '\t'; break;
                case 'r':  s += '\r'; break;
                case 'b':  s += '\b'; break;
                case 'f':  s += '\f'; break;
                case 'u': {
                    if (p + 4 < end) {
                        char hex[5] = {p[1], p[2], p[3], p[4], 0};
                        unsigned cp = (unsigned)std::strtoul(hex, nullptr, 16);
                        if (cp < 0x80) {
                            s += (char)cp;
                        } else if (cp < 0x800) {
                            s += (char)(0xC0 | (cp >> 6));
                            s += (char)(0x80 | (cp & 0x3F));
                        } else {
                            s += (char)(0xE0 | (cp >> 12));
                            s += (char)(0x80 | ((cp >> 6) & 0x3F));
                            s += (char)(0x80 | (cp & 0x3F));
                        }
                        p += 4;
                    }
                    break;
                }
                default: s += *p;
            }
        } else {
            s += *p;
        }
        ++p;
    }
    if (p < end) ++p;
    return s;
}

inline JsonValue parse_value(const char*& p, const char* end);

inline JsonValue parse_object(const char*& p, const char* end) {
    JsonValue v;
    v.type = JsonValue::J_OBJECT;
    v.obj_ptr = std::make_unique<std::unordered_map<std::string, JsonValue>>();
    ++p;
    skip_ws(p, end);
    if (p < end && *p == '}') { ++p; return v; }
    while (p < end) {
        skip_ws(p, end);
        std::string key = parse_string(p, end);
        skip_ws(p, end);
        if (p < end && *p == ':') ++p;
        skip_ws(p, end);
        v.obj_ptr->emplace(std::move(key), parse_value(p, end));
        skip_ws(p, end);
        if (p >= end || *p != ',') break;
        ++p;
    }
    if (p < end && *p == '}') ++p;
    return v;
}

inline JsonValue parse_array(const char*& p, const char* end) {
    JsonValue v;
    v.type = JsonValue::J_ARRAY;
    v.arr_ptr = std::make_unique<std::vector<JsonValue>>();
    ++p;
    skip_ws(p, end);
    if (p < end && *p == ']') { ++p; return v; }
    while (p < end) {
        skip_ws(p, end);
        v.arr_ptr->push_back(parse_value(p, end));
        skip_ws(p, end);
        if (p >= end || *p != ',') break;
        ++p;
    }
    if (p < end && *p == ']') ++p;
    return v;
}

inline JsonValue parse_value(const char*& p, const char* end) {
    skip_ws(p, end);
    if (p >= end) return {};
    if (*p == '{') return parse_object(p, end);
    if (*p == '[') return parse_array(p, end);
    if (*p == '"') {
        JsonValue v;
        v.type = JsonValue::J_STRING;
        v.str_val = parse_string(p, end);
        return v;
    }
    if (*p == 't' && p + 3 < end) {
        p += 4;
        JsonValue v;
        v.type = JsonValue::J_BOOL;
        v.bool_val = true;
        return v;
    }
    if (*p == 'f' && p + 4 < end) {
        p += 5;
        JsonValue v;
        v.type = JsonValue::J_BOOL;
        v.bool_val = false;
        return v;
    }
    if (*p == 'n' && p + 3 < end) {
        p += 4;
        return {};
    }
    JsonValue v;
    v.type = JsonValue::J_NUMBER;
    char* num_end;
    v.num_val = std::strtod(p, &num_end);
    p = num_end;
    return v;
}

} // namespace json_detail

inline JsonValue json_parse(const char* data, size_t len) {
    const char* p = data;
    const char* end = data + len;
    return json_detail::parse_value(p, end);
}

inline JsonValue json_parse(const std::string& s) {
    return json_parse(s.data(), s.size());
}
