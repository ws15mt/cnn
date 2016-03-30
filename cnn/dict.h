#ifndef CNN_DICT_H_
#define CNN_DICT_H_

#include <cassert>
#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <typeinfo>
#include <cnn/macros.h>
#include <boost/version.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/unordered_map.hpp>
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

template<class T>
class stId2String {
    typedef std::unordered_map<int, T> I2TMap;
    typedef std::unordered_map<T, int> T2IMap;
    std::vector<int> logicId2phyId; /// logic id is in sequence order

public:
    stId2String() : frozen(false) {
    }

    std::unordered_map<int, int> phyId2logicId; /// physical id to logic id
    int phyIdOflogicId(int logicid) { if (logicid >= 0 && logicid < logicId2phyId.size()) return logicId2phyId[logicid]; return -1; }

    inline unsigned size() const { return words_.size(); }

    inline bool Contains(const T& words) {
        return !(words_.find(words) == words_.end());
    }

    inline bool Contains(const int& id) {
        return !(d_.find(id) == d_.end());
    }

    void Freeze() { frozen = true; }

    inline int Convert(const int & id, const T& word)
    {
        auto i = words_.find(word);
        if (i == words_.end()) {
            if (frozen)
                return -1; 
            words_[word] = id; 
            d_[id] = word; 
            logicId2phyId.push_back(id); 
            phyId2logicId[id] = logicId2phyId.size() - 1; 
        }
        return id;
    }

    inline T Convert(const int& id)
    {
        if (d_.find(id) != d_.end())
            return d_[id];
        else
            return boost::lexical_cast<T>("");
    }

    void Clear() { words_.clear(); d_.clear(); }
private:
    bool frozen;
    I2TMap d_;
    T2IMap words_;

    friend class boost::serialization::access;
#if BOOST_VERSION >= 105600
    template<class Archive> void serialize(Archive& ar, const unsigned int) {
        ar & d_;
        ar & words_;
        ar & frozen;
        ar & logicId2phyId;
        ar & phyId2logicId;
    }
#else
    template<class Archive> void serialize(Archive& ar, const unsigned int) {
        throw std::invalid_argument("Serializing dictionaries is only supported on versions of boost 1.56 or higher");
    }
#endif
};

/// these are for word, either string or its index, to its tfidf weight
typedef boost::unordered::unordered_map<std::string, cnn::real> tWord2TfIdf;
typedef boost::unordered::unordered_map<int, cnn::real> tWordid2TfIdf;


} // namespace cnn


#endif
