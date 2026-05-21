#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "binparse::binparse" for configuration "Release"
set_property(TARGET binparse::binparse APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(binparse::binparse PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libbinparse.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS binparse::binparse )
list(APPEND _IMPORT_CHECK_FILES_FOR_binparse::binparse "${_IMPORT_PREFIX}/lib/libbinparse.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
