#ifndef CNN_DICT_H_
#define CNN_DICT_H_

#include <cassert>
#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <typeinfo>

#include <boost/version.hpp>
#include <boost/algorithm/string.hpp>
#if BOOST_VERSION >= 105600
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/string.hpp>
#endif

//#define INPUT_UTF8

using namespace std;
namespace cnn {

template<class T>
class stDict {
    T s_unk; 
 typedef std::unordered_map<T, int> Map;
 public:
  stDict() : frozen(false) {
#ifdef INPUT_UTF8
      s_unk = L"<unk>";
#else
      s_unk = "<unk>";
#endif
  }

  inline unsigned size() const { return words_.size(); }

  inline bool Contains(const T& words) {
      return !(d_.find(words) == d_.end());
  }

  void Freeze() { frozen = true; }

  inline int Convert(const T& word, bool backofftounk = false)
  {
    auto i = d_.find(word);
    if (i == d_.end()) {
      if (frozen) {
          if (backofftounk && d_.find(s_unk) != d_.end())
          {
              return d_[s_unk];
          }
          else
          {
#ifdef INPUT_UTF8
              std::wcerr << L"Unknown word encountered: " << std::endl;
#else
              std::cerr << "Unknown word encountered: " << std::endl;
#endif
              throw std::runtime_error("Unknown word encountered in frozen dictionary: ");
          }
      }
      words_.push_back(word);
      return d_[word] = words_.size() - 1;
    } else {
      return i->second;
    }
  }

  inline const T& Convert(const int& id) const {
      assert(id < (int)words_.size());
      return words_[id];
  }

  void SetUnk(const std::string& word) {
    if (!frozen)
      throw std::runtime_error("Please call SetUnk() only after dictionary is frozen");
    if (map_unk)
      throw std::runtime_error("Set UNK more than one time");
  
    // temporarily unfrozen the dictionary to allow the add of the UNK
    frozen = false;
    unk_id = Convert(word);
    frozen = true;
  
    map_unk = true;
  }

  void Clear() { words_.clear(); d_.clear();  }

  std::vector<T> GetWordList() { return words_; };

 private:
  bool frozen;
  bool map_unk; // if true, map unknown word to unk_id
  int unk_id; 
  std::vector<T> words_;
  Map d_;

  friend class boost::serialization::access;
#if BOOST_VERSION >= 105600
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & frozen;
    ar & map_unk;
    ar & unk_id;
    ar & words_;
    ar & d_;
  }
#else
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    throw std::invalid_argument("Serializing dictionaries is only supported on versions of boost 1.56 or higher");
  }
#endif
};

typedef stDict<std::string> Dict;
typedef stDict<std::wstring> WDict;

std::vector<int> ReadSentence(const std::string& line, Dict* sd);
std::vector<int> ReadSentence(const std::string& line, WDict* sd);
void ReadSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td);
void ReadSentencePair(const std::string& line, std::vector<int>* s, WDict* sd, std::vector<int>* t, WDict* td);

} // namespace cnn


#endif
