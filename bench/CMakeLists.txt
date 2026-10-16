# Standalone kernel benchmark harness. Reached only when NERT_BUILD_BENCH=ON.
#
# Bench tests must be registrable on their own: the root calls enable_testing()
# only when the runtime tests are enabled, so make sure it has happened here.
if(NOT CMAKE_TESTING_ENABLED)
    enable_testing()
endif()

add_subdirectory(utils)
