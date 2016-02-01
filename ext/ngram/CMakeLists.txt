# ########## cnn library ##########
# Sources:

# Headers:
set(ngram_HDRS
    ngram.h
)


# actual target:
if(WITH_CUDA_BACKEND)
add_library(ngram STATIC ${ngram_HDRS} ${LIBS})
else()
add_library(ngram ${ngram_HDRS} ${LIBS})
endif(WITH_CUDA_BACKEND)

install(FILES ${ngram_HDRS} DESTINATION include/cnn)
install(TARGETS ngram DESTINATION lib)

# target_compile_features(cnn PRIVATE cxx_range_for)

