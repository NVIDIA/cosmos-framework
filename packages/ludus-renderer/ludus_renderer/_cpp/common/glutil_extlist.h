// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef GL_VERSION_1_2
GLUTIL_EXT(void,   glTexImage3D,                GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels);
#endif
#ifndef GL_VERSION_1_5
GLUTIL_EXT(void,   glBindBuffer,                GLenum target, GLuint buffer);
GLUTIL_EXT(void,   glBufferData,                GLenum target, ptrdiff_t size, const void* data, GLenum usage);
GLUTIL_EXT(void,   glGenBuffers,                GLsizei n, GLuint* buffers);
#endif
#ifndef GL_VERSION_2_0
GLUTIL_EXT(void,   glAttachShader,              GLuint program, GLuint shader);
GLUTIL_EXT(void,   glCompileShader,             GLuint shader);
GLUTIL_EXT(GLuint, glCreateProgram,             void);
GLUTIL_EXT(GLuint, glCreateShader,              GLenum type);
GLUTIL_EXT(void,   glDrawBuffers,               GLsizei n, const GLenum* bufs);
GLUTIL_EXT(void,   glEnableVertexAttribArray,   GLuint index);
GLUTIL_EXT(void,   glGetProgramInfoLog,         GLuint program, GLsizei bufSize, GLsizei* length, char* infoLog);
GLUTIL_EXT(void,   glGetProgramiv,              GLuint program, GLenum pname, GLint* param);
GLUTIL_EXT(GLint,  glGetUniformLocation,        GLuint program, const char* name);
GLUTIL_EXT(void,   glLinkProgram,               GLuint program);
GLUTIL_EXT(void,   glShaderSource,              GLuint shader, GLsizei count, const char *const* string, const GLint* length);
GLUTIL_EXT(void,   glUniform1f,                 GLint location, GLfloat v0);
GLUTIL_EXT(void,   glUniform2f,                 GLint location, GLfloat v0, GLfloat v1);
GLUTIL_EXT(void,   glUseProgram,                GLuint program);
GLUTIL_EXT(void,   glVertexAttribPointer,       GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer);
#endif
#ifndef GL_VERSION_3_1
GLUTIL_EXT(void,   glBindBufferBase,            GLenum target, GLuint index, GLuint buffer);
GLUTIL_EXT(GLuint, glGetUniformBlockIndex,      GLuint program, const char* uniformBlockName);
GLUTIL_EXT(void,   glUniformBlockBinding,       GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding);
#endif
#ifndef GL_VERSION_3_2
GLUTIL_EXT(void,   glFramebufferTexture,        GLenum target, GLenum attachment, GLuint texture, GLint level);
#endif
#ifndef GL_ARB_framebuffer_object
GLUTIL_EXT(void,   glBindFramebuffer,           GLenum target, GLuint framebuffer);
GLUTIL_EXT(void,   glGenFramebuffers,           GLsizei n, GLuint* framebuffers);
#endif
#ifndef GL_ARB_vertex_array_object
GLUTIL_EXT(void,   glBindVertexArray,           GLuint array);
GLUTIL_EXT(void,   glGenVertexArrays,           GLsizei n, GLuint* arrays);
#endif
#ifndef GL_ARB_multi_draw_indirect
GLUTIL_EXT(void,   glMultiDrawElementsIndirect, GLenum mode, GLenum type, const void *indirect, GLsizei primcount, GLsizei stride);
#endif

// Additional functions for shader compilation and mesh shaders
#ifndef GL_VERSION_2_0_EXTRA
GLUTIL_EXT(void,   glGetShaderiv,               GLuint shader, GLenum pname, GLint* params);
GLUTIL_EXT(void,   glGetShaderInfoLog,          GLuint shader, GLsizei bufSize, GLsizei* length, char* infoLog);
GLUTIL_EXT(void,   glUniform1i,                 GLint location, GLint v0);
GLUTIL_EXT(void,   glUniform1ui,                GLint location, GLuint v0);
GLUTIL_EXT(void,   glDeleteShader,              GLuint shader);
GLUTIL_EXT(void,   glDeleteProgram,             GLuint program);
#endif

// Mesh shader extension (GL_NV_mesh_shader)
#ifndef GL_NV_mesh_shader
GLUTIL_EXT(void,   glDrawMeshTasksNV,           GLuint first, GLuint count);
GLUTIL_EXT(void,   glDrawMeshTasksIndirectNV,   ptrdiff_t indirect);
#endif

// MSAA support
#ifndef GL_ARB_texture_multisample
GLUTIL_EXT(void,   glTexImage3DMultisample,     GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations);
#endif
#ifndef GL_ARB_framebuffer_object_EXTRA
GLUTIL_EXT(void,   glBlitFramebuffer,           GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
GLUTIL_EXT(GLenum, glCheckFramebufferStatus,    GLenum target);
GLUTIL_EXT(void,   glDeleteFramebuffers,        GLsizei n, const GLuint* framebuffers);
GLUTIL_EXT(void,   glFramebufferTextureLayer,   GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer);
#endif

// GL sync / fence functions
#ifndef GL_ARB_sync
GLUTIL_EXT(GLsync, glFenceSync,              GLenum condition, GLbitfield flags);
GLUTIL_EXT(GLenum, glClientWaitSync,         GLsync sync, GLbitfield flags, GLuint64 timeout);
GLUTIL_EXT(void,   glDeleteSync,             GLsync sync);
#endif

//------------------------------------------------------------------------
