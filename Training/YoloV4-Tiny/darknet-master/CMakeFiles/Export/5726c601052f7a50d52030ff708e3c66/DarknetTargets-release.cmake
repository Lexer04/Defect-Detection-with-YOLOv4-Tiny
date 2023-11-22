#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Darknet::dark" for configuration "Release"
set_property(TARGET Darknet::dark APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Darknet::dark PROPERTIES
  IMPORTED_IMPLIB_RELEASE "C:/Users/yoshu/Documents/Materi Kuliah/TA/darknet-master/darknet.lib"
  IMPORTED_LOCATION_RELEASE "C:/Users/yoshu/Documents/Materi Kuliah/TA/darknet-master/darknet.dll"
  )

list(APPEND _cmake_import_check_targets Darknet::dark )
list(APPEND _cmake_import_check_files_for_Darknet::dark "C:/Users/yoshu/Documents/Materi Kuliah/TA/darknet-master/darknet.lib" "C:/Users/yoshu/Documents/Materi Kuliah/TA/darknet-master/darknet.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
