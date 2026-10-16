cmake_minimum_required(VERSION 3.16)
project(dead_orbit C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Use a system raylib 5.5 if one is installed, otherwise fetch it once from GitHub.
# For a fully offline build, clone raylib 5.5 somewhere and pass
#   -DFETCHCONTENT_SOURCE_DIR_RAYLIB=/path/to/raylib
find_package(raylib 5.5 QUIET)
if(NOT raylib_FOUND)
  include(FetchContent)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
  set(OPENGL_VERSION "3.3" CACHE STRING "" FORCE)
  FetchContent_Declare(raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG 5.5
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(raylib)
endif()

add_executable(dead-orbit
  src/main.c
  src/gfx.c
  src/world.c
  src/audio.c
  src/ui.c
  src/settings.c)
target_link_libraries(dead-orbit PRIVATE raylib m)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(dead-orbit PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
endif()

add_custom_command(TARGET dead-orbit POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:dead-orbit>/assets)

include(GNUInstallDirs)
install(TARGETS dead-orbit RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY assets DESTINATION ${CMAKE_INSTALL_DATADIR}/dead-orbit)
install(FILES dead-orbit.desktop DESTINATION ${CMAKE_INSTALL_DATADIR}/applications)
