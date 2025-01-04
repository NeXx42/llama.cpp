set(TARGET LLMPipelineServer)

add_executable(${TARGET} LLMPipelineServer.cpp SessionManager.cpp Session.cpp)

install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(ModelTypes)