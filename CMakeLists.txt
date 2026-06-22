cmake_minimum_required(VERSION 3.20)
project(llm-collab
    VERSION 2.1.0
    DESCRIPTION "Polyglot Neurosynaptic LLM Orchestration — C++ + Fortran + Python"
    LANGUAGES CXX Fortran
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_Fortran_STANDARD 2018)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)

# ─── Fortran: Parallel HD/SDM Encoding (OpenMP) ───────────
set(FORTRAN_SOURCES
    engine/hd_encode.f90
    engine/parallel_ops.f90
)
add_library(synapse_encode SHARED ${FORTRAN_SOURCES})
target_link_libraries(synapse_encode PRIVATE OpenMP::OpenMP_Fortran)
set_target_properties(synapse_encode PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# ─── C++: High-Performance Routing Engine ─────────────────
if(pybind11_FOUND)
    set(ROUTER_SOURCES
        engine/router_cpp/src/router.cpp
        engine/router_cpp/src/pathway.cpp
        engine/router_cpp/src/thread_pool.cpp
        engine/router_cpp/src/bindings.cpp
    )
    add_library(synapse_router SHARED ${ROUTER_SOURCES})
    target_include_directories(synapse_router PRIVATE
        engine/router_cpp/include
        ${Python_INCLUDE_DIRS}
    )
    target_link_libraries(synapse_router PRIVATE
        pybind11::module
        ${Python_LIBRARIES}
        synapse_encode
    )
    set_target_properties(synapse_router PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
endif()
