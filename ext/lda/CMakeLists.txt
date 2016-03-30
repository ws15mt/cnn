# ########## cnn library ##########
# Sources:

# Headers:
set(lda_HDRS
    lda.h
)


# actual target:
if(WITH_CUDA_BACKEND)
add_library(lda STATIC ${lda_HDRS} ${LIBS})
else()
add_library(lda ${lda_HDRS} ${LIBS})
endif(WITH_CUDA_BACKEND)

#add_library(dialogue SHARED ${dialogue_library_HDRS} ${LIBS})

install(FILES ${lda_HDRS} DESTINATION include/cnn)
install(TARGETS lda DESTINATION lib)

# target_compile_features(cnn PRIVATE cxx_range_for)

