cmake_minimum_required(VERSION 3.20)
project(TB_Rootifier VERSION 0.3.0 LANGUAGES CXX)

include(FetchContent)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(ROOT_DIR "/home/baschmid/sw/cern_root/root_install" CACHE PATH "Path to the ROOT installation prefix")
list(APPEND CMAKE_PREFIX_PATH ${ROOT_DIR})
find_package(ROOT REQUIRED COMPONENTS
  Core RIO Hist Tree TreePlayer MathCore Graf Gpad Rint Thread 
)

set(ROOT_RUNTIME_PREFIX ${ROOT_DIR})
set(ROOT_RUNTIME_LIBDIR ${ROOT_RUNTIME_PREFIX}/lib)
if(NOT EXISTS ${ROOT_RUNTIME_LIBDIR})
  get_filename_component(ROOT_RUNTIME_PREFIX ${ROOT_RUNTIME_PREFIX} DIRECTORY)
  set(ROOT_RUNTIME_LIBDIR ${ROOT_RUNTIME_PREFIX}/lib)
endif()
if(NOT EXISTS ${ROOT_RUNTIME_LIBDIR})
  get_target_property(ROOT_CORE_LIBRARY ROOT::Core IMPORTED_LOCATION)
  if(ROOT_CORE_LIBRARY)
    get_filename_component(ROOT_RUNTIME_LIBDIR ${ROOT_CORE_LIBRARY} DIRECTORY)
    get_filename_component(ROOT_RUNTIME_PREFIX ${ROOT_RUNTIME_LIBDIR} DIRECTORY)
  else()
    message(FATAL_ERROR "Unable to determine the ROOT runtime location for portable packaging; set ROOT_DIR to the ROOT installation prefix")
  endif()
endif()

find_package(Threads REQUIRED)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/scripts)

set(SCRIPTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rootifier)
file(GLOB SCRIPT_SOURCES CONFIGURE_DEPENDS
  ${SCRIPTS_DIR}/*.cxx
  ${SCRIPTS_DIR}/*.cpp
)
list(FILTER SCRIPT_SOURCES EXCLUDE REGEX "/\\._[^/]+\\.(cxx|cpp)$")


# ---- cxxopts via FetchContent ----
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG        v3.2.0
)
FetchContent_GetProperties(cxxopts)
if(NOT cxxopts_POPULATED)
  FetchContent_Populate(cxxopts)
  add_subdirectory(${cxxopts_SOURCE_DIR} ${cxxopts_BINARY_DIR})
endif()

# ---- spdlog via FetchContent ----
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.14.1
)
FetchContent_MakeAvailable(spdlog)

# ---- binparse via FetchContent (your repo must export binparse::binparse) ----
FetchContent_Declare(
  binparse
  GIT_REPOSITORY https://github.com/RockPie/CRU_RDH_Decoder.git
  GIT_TAG        v0.5
)
FetchContent_MakeAvailable(binparse)

# --- Fetch nlohmann/json ---
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3   # or any stable release
)

FetchContent_MakeAvailable(nlohmann_json)

# --- Fetch Eigen ---
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)

FetchContent_MakeAvailable(eigen)

set(SCRIPT_TARGETS)
foreach(src IN LISTS SCRIPT_SOURCES)
  get_filename_component(exec_name ${src} NAME_WE)
  add_executable(${exec_name} ${src})

  target_include_directories(${exec_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  # add include path
  target_include_directories(${exec_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)

  target_link_libraries(${exec_name}
    PRIVATE
        cxxopts
        spdlog::spdlog
        binparse::binparse
        nlohmann_json::nlohmann_json
        Eigen3::Eigen
        ROOT::Core ROOT::RIO ROOT::Hist ROOT::Tree ROOT::MathCore
        ROOT::Graf ROOT::Gpad
        Threads::Threads
        ROOT::RHTTP
  )

  target_compile_definitions(${exec_name} PRIVATE STAGE_NAME="${exec_name}")

  list(APPEND SCRIPT_TARGETS ${exec_name})
endforeach()

if(SCRIPT_TARGETS)
  add_custom_target(all_scripts DEPENDS ${SCRIPT_TARGETS})
endif()

if(SCRIPT_TARGETS)
  install(TARGETS ${SCRIPT_TARGETS}
    RUNTIME DESTINATION libexec
    COMPONENT portable_runtime
  )
endif()

if(APPLE AND SCRIPT_TARGETS)
  set(PORTABLE_BUNDLE_NAME "TB_Rootifier-macos")
  set(PORTABLE_DIST_DIR "${CMAKE_BINARY_DIR}/dist/${PORTABLE_BUNDLE_NAME}")
  set(PORTABLE_WRAPPER_DIR "${CMAKE_CURRENT_BINARY_DIR}/portable_wrappers")
  file(MAKE_DIRECTORY ${PORTABLE_WRAPPER_DIR})

  foreach(exec_name IN LISTS SCRIPT_TARGETS)
    set(PORTABLE_EXECUTABLE_NAME ${exec_name})
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/root_tool_wrapper.sh.in
      ${PORTABLE_WRAPPER_DIR}/${exec_name}
      @ONLY
    )
    install(PROGRAMS ${PORTABLE_WRAPPER_DIR}/${exec_name}
      DESTINATION bin
      COMPONENT portable_runtime
    )
  endforeach()

  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/Readme.md
    DESTINATION .
    COMPONENT portable_runtime
    OPTIONAL
  )

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/macos_portable_bundle.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/macos_portable_bundle.cmake
    @ONLY
  )

  add_custom_target(package_portable_macos
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${PORTABLE_DIST_DIR}
    COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR} --prefix ${PORTABLE_DIST_DIR} --component portable_runtime
    COMMAND ${CMAKE_COMMAND} -DPORTABLE_DIST_DIR=${PORTABLE_DIST_DIR} -P ${CMAKE_CURRENT_BINARY_DIR}/macos_portable_bundle.cmake
    DEPENDS ${SCRIPT_TARGETS}
    COMMENT "Building a relocatable macOS distribution with bundled ROOT runtime"
    VERBATIM
  )
endif()

message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
if(APPLE)
  message(STATUS "macOS deployment target: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()


