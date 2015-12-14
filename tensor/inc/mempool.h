#ifndef MEMPOOL_H
#define MEMPOOL_H
/*!
 * \file mempool.h
 * \brief A very simple memory pool for both CPU and GPU
 * 		CPU memory malloc and free is very fast, but GPU not.
 * 		GPU memory malloc is very slow, so I think we need some 'layer' to manage memory by our hand.
 * \author qhduan.com
 * \date 2014-05-13
 */

#include <map>
#include <unordered_map>
#include <tuple>
#include <utility>
#include "../inc/mem.h"

namespace std {

template <>
struct hash<pair<int,int> > {
  size_t operator() (const pair<int,int>& p) const {
    size_t h1 = hash<int>()(p.first);
    size_t h2 = hash<int>()(p.second);
    return h1 ^ (h2 << 1);
  }
};

}

namespace tensor {


/*!
 * \brief Memory pool class
 */
template <Device dev>
class Mempool {
public:

  /*!
   * \brief return singleton Mempool instance
   */
  static Mempool& instance () {
    if (Mempool<dev>::instance_ == NULL) {
      Mempool<dev>::instance_ = new Mempool();
    }
    return *Mempool<dev>::instance_;
  }

	/*!
	 * \brief malloc memory
	 * \param size how many element you need, not in byte
	 * \return pointer to result
	 */
	template <typename T>
	T* malloc (int size);

	/*!
	 * \brief malloc 2d memory
	 * \param height height of you want, not in byte
   * \param width width of you want, not in bytes
   * \param pitch return pitch, infact it's kinda the true height, pitch >= height
   * element = w * pitch + h, which 0 <= w <= width, 0 <= h <= height
	 * \return pointer to result
	 */
	// template <typename T>
	// T* malloc (int height, int width, int* pitch);

	/*!
	 * \brief free memroy
	 * \param ptr pointer to Mempool malloc
	 */
	template <typename T>
	void free (T* ptr);


  void free_idle ();

private:

	/*! \brief construct singleton */
	Mempool ();

	/*! \brief destroy singleton */
	~Mempool ();

  /*!
   * \brief store the pointers which just idle (store freed pointer)
   * key: height, width, for 1D malloc, height = length, width = 1
   * value: pointer to area, pitch of area, for 1D malloc, pitch = 0
   */
  //std::multimap<std::pair<int, int>, std::pair<void*, int> > idle_;
  std::unordered_multimap<std::pair<int, int>, std::pair<void*, int> > idle_;

  /*!
   * \brief store all pointers
   * key: pointer to area
   * value: height, width, pitch
   */
  //std::map<void*, std::tuple<int, int, int> > index_;
  std::unordered_map<void*, std::tuple<int, int, int> > index_;

	/*! \brief how many pointer Mempool 'real' malloc */
	int malloc_time_;

	/*! \brief how many pointer Mempool 're' malloc */
	int re_malloc_time_;

  static Mempool<dev>* instance_;
};

}

#endif // MEMPOOL_H
