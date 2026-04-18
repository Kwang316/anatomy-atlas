cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)
project(AnatomyAtlas)

set(EXTENSION_HOMEPAGE "https://github.com/Kwang316/anatomy-atlas")
set(EXTENSION_CATEGORY "Education")
set(EXTENSION_CONTRIBUTORS "Ke Wang (Medical Imaging Engineer)")
set(EXTENSION_DESCRIPTION "Interactive 3D Anatomy Atlas — explore organs by system, click to identify, quiz mode.")
set(EXTENSION_DEPENDS "NA")

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

add_subdirectory(AnatomyAtlas)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
