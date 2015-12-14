#ifndef MEM_H
#define MEM_H
/*!
 * \file mem.h
 * \brief contain memory operation
 * \author ner.center
 * \date 2014-06-21
 */

#include "../inc/device.hpp"

namespace tensor {

/*!
 * \brief cross-platform malloc and free
 * There're two reason we add this file instead of using library's function directly:
 * 1. We could make malloc of each platform could throw std::bad_alloc
 * 2. Mempool don't touch cuda, so it could use traits of c++ 11 (eg. hash table)
 */
template <Device dev>
class Mem {
public:
	/*!
	 * \brief instead system malloc
	 * \param bytes you want
	 * \return pointer to the area
	 */
	static void* malloc(int bytes);

  /*!
   * \brief mallo a 2D, beware: column major!!!
   * 1. cudaMallocPitch is kinda row major, but cuBlas need column major
   *    we only use column major
   * 2. cpu algorithm doesn't need pitch, but for test and same behavior
   *    as gpu, we use at least 128bits of pitch of cpu.
   *    eg. height = 50bytes, we return 128bytes of pitch, height = 129 bytes
   *    we return 256bytes of pitch
   * \param pitch return the pitch
   * \param hbytes height in bytes
   * \param wbytes width in bytes
   * \return pointer to the area
   */
  static void* malloc(int hbytes, int wbytes, int* pitch);

	/*!
	 * \brief instead system free
	 * \param pointer want free
	 */
	static void free(void* ptr);
};

}

#endif // MEM_H
