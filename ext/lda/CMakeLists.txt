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
if(WITH_CUDA_BACKEND)
  set(CUDA_SEPARABLE_COMPILATION ON)
  list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_20,code=sm_20;-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-std=c++11;-O2;-DVERBOSE")
  SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
  cuda_add_library(cnncuda STATIC gpu-ops.cu)
endif(WITH_CUDA_BACKEND)

install(FILES ${lda_HDRS} DESTINATION include/cnn)
install(TARGETS lda DESTINATION lib)

# target_compile_features(cnn PRIVATE cxx_range_for)

