# ########## cnn library ##########
# Sources:

# Headers:
set(trainer_library_HDRS
    train_proc.h
    train_proc_wrapper.h
    eval_proc.h
)


# actual target:
if(WITH_CUDA_BACKEND)
add_library(trainproc STATIC ${trainer_library_HDRS} ${LIBS})
else()
add_library(trainproc ${trainer_library_HDRS} ${LIBS})
endif(WITH_CUDA_BACKEND)

#add_library(trainproc SHARED ${cnn_library_SRCS} ${cnn_library_HDRS} ${LIBS})

install(FILES ${trainer_library_HDRS} DESTINATION include/cnn)
install(TARGETS trainproc DESTINATION lib)

# target_compile_features(cnn PRIVATE cxx_range_for)

