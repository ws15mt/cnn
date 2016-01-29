# ########## cnn library ##########
# Sources:

# Headers:
set(enc_dec_library_HDRS
    encdec.h
)


# actual target:
if(WITH_CUDA_BACKEND)
add_library(encdec STATIC ${enc_dec_library_HDRS} ${LIBS})
else()
add_library(encdec ${enc_dec_library_HDRS} ${LIBS})
endif(WITH_CUDA_BACKEND)

#add_library(encdec SHARED ${enc_dec_library_SRCS} ${enc_dec_library_HDRS} ${LIBS})

install(FILES ${enc_dec_library_HDRS} DESTINATION include/cnn)
install(TARGETS encdec DESTINATION lib)

# target_compile_features(cnn PRIVATE cxx_range_for)

