#ifndef _TIMING_H_
#define _TIMING_H_

#include <iostream>
#include <string>
#include <chrono>

namespace cnn {

struct Timer {
  Timer(const std::string& msg) : msg(msg), start(std::chrono::high_resolution_clock::now()) {}
  ~Timer() {
    auto stop = std::chrono::high_resolution_clock::now();
    std::cerr << '[' << msg << ' ' << std::chrono::duration<double, std::milli>(stop-start).count() << " ms]\n";
  }
  void WordsPerSecond(int nwords)
  {
      auto stop = std::chrono::high_resolution_clock::now();
      int ms = std::chrono::duration<double, std::milli>(stop - start).count();
      cnn::real se = ms * 1000.0;
      cnn::real wps = nwords / se;
      std::cerr << "[ words per second ] = " << wps << std::endl; 
  }
  std::string msg;
  std::chrono::high_resolution_clock::time_point start;
};

} // namespace cnn

#endif
