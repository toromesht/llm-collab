cmake_minimum_required(VERSION 3.20)
project(llm-collab
    VERSION 2.1.0
    DESCRIPTION "Polyglot Neurosynaptic LLM Orchestration — C++ + Fortran + Python"
    LANGUAGES CXX Fortran
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_Fortran_STANDARD 2018)

# Find OpenMP FIRST (needed by Fortran)
find_package(OpenMP REQUIRED)

# Find correct Python (must be 64-bit, matching the compiler)
find_package(Python 3.11 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)

# ─── Fortran: Parallel HD/SDM Encoding (OpenMP) ───────────
# STATIC library to avoid MinGW c_f_pointer_ DLL export issue
set(FORTRAN_SOURCES
    engine/hd_encode.f90
    engine/parallel_ops.f90
    engine/brainstem.f90
    engine/brainstem_cli.f90
)
add_library(synapse_encode STATIC ${FORTRAN_SOURCES})
target_link_libraries(synapse_encode PRIVATE OpenMP::OpenMP_Fortran)
target_compile_options(synapse_encode PRIVATE -O2 -fopenmp)
set_target_properties(synapse_encode PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    POSITION_INDEPENDENT_CODE ON
)

# ─── C++: High-Performance Routing Engine ─────────────────
if(pybind11_FOUND)
    set(ROUTER_SOURCES
        engine/router_cpp/src/router.cpp
        engine/router_cpp/src/pathway.cpp
        engine/router_cpp/src/thread_pool.cpp
        engine/router_cpp/src/bindings.cpp
    )
    pybind11_add_module(synapse_router ${ROUTER_SOURCES})
    target_include_directories(synapse_router PRIVATE
        engine/router_cpp/include
    )
    target_compile_options(synapse_router PRIVATE -O2)
    set_target_properties(synapse_router PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )

    # Copy MinGW runtime DLLs alongside the Python module
    # so it's self-contained — no PATH manipulation needed
    if(MINGW AND WIN32)
        # Find the compiler's bin directory from CMAKE_CXX_COMPILER
        get_filename_component(GXX_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
        file(GLOB RUNTIME_DLLS
            "${GXX_BIN_DIR}/libgcc_s_seh-1.dll"
            "${GXX_BIN_DIR}/libstdc++-6.dll"
            "${GXX_BIN_DIR}/libwinpthread-1.dll"
        )
        foreach(DLL ${RUNTIME_DLLS})
            add_custom_command(TARGET synapse_router POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DLL}" "${CMAKE_BINARY_DIR}/lib/"
                COMMENT "Copying runtime DLL: ${DLL}"
            )
        endforeach()
    endif()
endif()
