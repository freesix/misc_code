# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/freesix/3DMax-Cliques/Linux

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/freesix/3DMax-Cliques/Linux/build

# Include any dependencies generated for this target.
include CMakeFiles/MAC.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/MAC.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/MAC.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MAC.dir/flags.make

CMakeFiles/MAC.dir/main.cpp.o: CMakeFiles/MAC.dir/flags.make
CMakeFiles/MAC.dir/main.cpp.o: /home/freesix/3DMax-Cliques/Linux/main.cpp
CMakeFiles/MAC.dir/main.cpp.o: CMakeFiles/MAC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/freesix/3DMax-Cliques/Linux/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/MAC.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/MAC.dir/main.cpp.o -MF CMakeFiles/MAC.dir/main.cpp.o.d -o CMakeFiles/MAC.dir/main.cpp.o -c /home/freesix/3DMax-Cliques/Linux/main.cpp

CMakeFiles/MAC.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MAC.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/freesix/3DMax-Cliques/Linux/main.cpp > CMakeFiles/MAC.dir/main.cpp.i

CMakeFiles/MAC.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MAC.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/freesix/3DMax-Cliques/Linux/main.cpp -o CMakeFiles/MAC.dir/main.cpp.s

CMakeFiles/MAC.dir/desc_dec.cpp.o: CMakeFiles/MAC.dir/flags.make
CMakeFiles/MAC.dir/desc_dec.cpp.o: /home/freesix/3DMax-Cliques/Linux/desc_dec.cpp
CMakeFiles/MAC.dir/desc_dec.cpp.o: CMakeFiles/MAC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/freesix/3DMax-Cliques/Linux/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/MAC.dir/desc_dec.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/MAC.dir/desc_dec.cpp.o -MF CMakeFiles/MAC.dir/desc_dec.cpp.o.d -o CMakeFiles/MAC.dir/desc_dec.cpp.o -c /home/freesix/3DMax-Cliques/Linux/desc_dec.cpp

CMakeFiles/MAC.dir/desc_dec.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MAC.dir/desc_dec.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/freesix/3DMax-Cliques/Linux/desc_dec.cpp > CMakeFiles/MAC.dir/desc_dec.cpp.i

CMakeFiles/MAC.dir/desc_dec.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MAC.dir/desc_dec.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/freesix/3DMax-Cliques/Linux/desc_dec.cpp -o CMakeFiles/MAC.dir/desc_dec.cpp.s

CMakeFiles/MAC.dir/funcs.cpp.o: CMakeFiles/MAC.dir/flags.make
CMakeFiles/MAC.dir/funcs.cpp.o: /home/freesix/3DMax-Cliques/Linux/funcs.cpp
CMakeFiles/MAC.dir/funcs.cpp.o: CMakeFiles/MAC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/freesix/3DMax-Cliques/Linux/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/MAC.dir/funcs.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/MAC.dir/funcs.cpp.o -MF CMakeFiles/MAC.dir/funcs.cpp.o.d -o CMakeFiles/MAC.dir/funcs.cpp.o -c /home/freesix/3DMax-Cliques/Linux/funcs.cpp

CMakeFiles/MAC.dir/funcs.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MAC.dir/funcs.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/freesix/3DMax-Cliques/Linux/funcs.cpp > CMakeFiles/MAC.dir/funcs.cpp.i

CMakeFiles/MAC.dir/funcs.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MAC.dir/funcs.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/freesix/3DMax-Cliques/Linux/funcs.cpp -o CMakeFiles/MAC.dir/funcs.cpp.s

CMakeFiles/MAC.dir/PCR.cpp.o: CMakeFiles/MAC.dir/flags.make
CMakeFiles/MAC.dir/PCR.cpp.o: /home/freesix/3DMax-Cliques/Linux/PCR.cpp
CMakeFiles/MAC.dir/PCR.cpp.o: CMakeFiles/MAC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/freesix/3DMax-Cliques/Linux/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/MAC.dir/PCR.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/MAC.dir/PCR.cpp.o -MF CMakeFiles/MAC.dir/PCR.cpp.o.d -o CMakeFiles/MAC.dir/PCR.cpp.o -c /home/freesix/3DMax-Cliques/Linux/PCR.cpp

CMakeFiles/MAC.dir/PCR.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MAC.dir/PCR.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/freesix/3DMax-Cliques/Linux/PCR.cpp > CMakeFiles/MAC.dir/PCR.cpp.i

CMakeFiles/MAC.dir/PCR.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MAC.dir/PCR.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/freesix/3DMax-Cliques/Linux/PCR.cpp -o CMakeFiles/MAC.dir/PCR.cpp.s

CMakeFiles/MAC.dir/registration.cpp.o: CMakeFiles/MAC.dir/flags.make
CMakeFiles/MAC.dir/registration.cpp.o: /home/freesix/3DMax-Cliques/Linux/registration.cpp
CMakeFiles/MAC.dir/registration.cpp.o: CMakeFiles/MAC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/freesix/3DMax-Cliques/Linux/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/MAC.dir/registration.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/MAC.dir/registration.cpp.o -MF CMakeFiles/MAC.dir/registration.cpp.o.d -o CMakeFiles/MAC.dir/registration.cpp.o -c /home/freesix/3DMax-Cliques/Linux/registration.cpp

CMakeFiles/MAC.dir/registration.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MAC.dir/registration.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/freesix/3DMax-Cliques/Linux/registration.cpp > CMakeFiles/MAC.dir/registration.cpp.i

CMakeFiles/MAC.dir/registration.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MAC.dir/registration.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/freesix/3DMax-Cliques/Linux/registration.cpp -o CMakeFiles/MAC.dir/registration.cpp.s

CMakeFiles/MAC.dir/visualization.cpp.o: CMakeFiles/MAC.dir/flags.make
CMakeFiles/MAC.dir/visualization.cpp.o: /home/freesix/3DMax-Cliques/Linux/visualization.cpp
CMakeFiles/MAC.dir/visualization.cpp.o: CMakeFiles/MAC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/freesix/3DMax-Cliques/Linux/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/MAC.dir/visualization.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/MAC.dir/visualization.cpp.o -MF CMakeFiles/MAC.dir/visualization.cpp.o.d -o CMakeFiles/MAC.dir/visualization.cpp.o -c /home/freesix/3DMax-Cliques/Linux/visualization.cpp

CMakeFiles/MAC.dir/visualization.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MAC.dir/visualization.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/freesix/3DMax-Cliques/Linux/visualization.cpp > CMakeFiles/MAC.dir/visualization.cpp.i

CMakeFiles/MAC.dir/visualization.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MAC.dir/visualization.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/freesix/3DMax-Cliques/Linux/visualization.cpp -o CMakeFiles/MAC.dir/visualization.cpp.s

# Object files for target MAC
MAC_OBJECTS = \
"CMakeFiles/MAC.dir/main.cpp.o" \
"CMakeFiles/MAC.dir/desc_dec.cpp.o" \
"CMakeFiles/MAC.dir/funcs.cpp.o" \
"CMakeFiles/MAC.dir/PCR.cpp.o" \
"CMakeFiles/MAC.dir/registration.cpp.o" \
"CMakeFiles/MAC.dir/visualization.cpp.o"

# External object files for target MAC
MAC_EXTERNAL_OBJECTS =

MAC: CMakeFiles/MAC.dir/main.cpp.o
MAC: CMakeFiles/MAC.dir/desc_dec.cpp.o
MAC: CMakeFiles/MAC.dir/funcs.cpp.o
MAC: CMakeFiles/MAC.dir/PCR.cpp.o
MAC: CMakeFiles/MAC.dir/registration.cpp.o
MAC: CMakeFiles/MAC.dir/visualization.cpp.o
MAC: CMakeFiles/MAC.dir/build.make
MAC: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
MAC: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
MAC: /usr/lib/x86_64-linux-gnu/libpcl_people.so
MAC: /usr/lib/libOpenNI.so
MAC: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
MAC: /usr/lib/x86_64-linux-gnu/libOpenNI2.so
MAC: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
MAC: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
MAC: /usr/local/lib/libigraph.a
MAC: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
MAC: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
MAC: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
MAC: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
MAC: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
MAC: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
MAC: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
MAC: /usr/lib/x86_64-linux-gnu/libpcl_features.so
MAC: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
MAC: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
MAC: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
MAC: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
MAC: /usr/lib/x86_64-linux-gnu/libpcl_search.so
MAC: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
MAC: /usr/lib/x86_64-linux-gnu/libpcl_io.so
MAC: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
MAC: /usr/lib/x86_64-linux-gnu/libpng.so
MAC: /usr/lib/x86_64-linux-gnu/libz.so
MAC: /usr/lib/libOpenNI.so
MAC: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
MAC: /usr/lib/x86_64-linux-gnu/libOpenNI2.so
MAC: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkInteractionImage-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
MAC: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQt-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkIOCore-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libfreetype.so
MAC: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkIOImage-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkRenderingUI-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libvtkkissfft-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libGLEW.so
MAC: /usr/lib/x86_64-linux-gnu/libX11.so
MAC: /usr/lib/x86_64-linux-gnu/libQt5OpenGL.so.5.15.3
MAC: /usr/lib/x86_64-linux-gnu/libQt5Widgets.so.5.15.3
MAC: /usr/lib/x86_64-linux-gnu/libQt5Gui.so.5.15.3
MAC: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5.15.3
MAC: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libtbb.so.12.5
MAC: /usr/lib/x86_64-linux-gnu/libvtksys-9.1.so.9.1.0
MAC: /usr/lib/x86_64-linux-gnu/libpcl_common.so
MAC: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.74.0
MAC: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.74.0
MAC: /usr/lib/x86_64-linux-gnu/libboost_date_time.so.1.74.0
MAC: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so.1.74.0
MAC: /usr/lib/x86_64-linux-gnu/libboost_serialization.so.1.74.0
MAC: /usr/lib/x86_64-linux-gnu/libqhull_r.so.8.0.2
MAC: /usr/lib/x86_64-linux-gnu/libm.so
MAC: /usr/lib/x86_64-linux-gnu/libarpack.so
MAC: /usr/lib/x86_64-linux-gnu/libblas.so
MAC: /usr/lib/x86_64-linux-gnu/libf77blas.so
MAC: /usr/lib/x86_64-linux-gnu/libatlas.so
MAC: /usr/lib/x86_64-linux-gnu/liblapack.so
MAC: /usr/lib/x86_64-linux-gnu/libblas.so
MAC: /usr/lib/x86_64-linux-gnu/libf77blas.so
MAC: /usr/lib/x86_64-linux-gnu/libatlas.so
MAC: /usr/lib/x86_64-linux-gnu/liblapack.so
MAC: /usr/lib/x86_64-linux-gnu/libxml2.so
MAC: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
MAC: /usr/lib/x86_64-linux-gnu/libpthread.a
MAC: CMakeFiles/MAC.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/freesix/3DMax-Cliques/Linux/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable MAC"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MAC.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/MAC.dir/build: MAC
.PHONY : CMakeFiles/MAC.dir/build

CMakeFiles/MAC.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MAC.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MAC.dir/clean

CMakeFiles/MAC.dir/depend:
	cd /home/freesix/3DMax-Cliques/Linux/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/freesix/3DMax-Cliques/Linux /home/freesix/3DMax-Cliques/Linux /home/freesix/3DMax-Cliques/Linux/build /home/freesix/3DMax-Cliques/Linux/build /home/freesix/3DMax-Cliques/Linux/build/CMakeFiles/MAC.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MAC.dir/depend
