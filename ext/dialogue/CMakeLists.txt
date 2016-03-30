# ########## cnn library ##########
# Sources:

# Headers:
set(dialogue_library_HDRS
    dialogue_process.h
    attention_with_intention.h
    dialogue.h
)


# actual target:
if(WITH_CUDA_BACKEND)
add_library(dialogue STATIC ${dialogue_library_HDRS} ${LIBS})
else()
add_library(dialogue ${dialogue_library_HDRS} ${LIBS})
endif(WITH_CUDA_BACKEND)

#add_library(dialogue SHARED ${dialogue_library_HDRS} ${LIBS})

install(FILES ${dialogue_library_HDRS} DESTINATION include/cnn)
install(TARGETS dialogue DESTINATION lib)

# target_compile_features(cnn PRIVATE cxx_range_for)

