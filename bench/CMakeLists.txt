# Shipped-kernel benchmark driver. Reached only when NERT_BUILD_BENCH=ON.
#
# Bench tests must be registrable on their own: the root calls enable_testing()
# only when the runtime tests are enabled, so make sure it has happened here.
if(NOT CMAKE_TESTING_ENABLED)
    enable_testing()
endif()

# Runtime-backed parity + timing: drives the shipped kernel tables, no
# vendored kernel copies. Device halves are gated on their own
# NERT_ENABLE_* flags (see bench/runtime/CMakeLists.txt).
add_subdirectory(runtime)
