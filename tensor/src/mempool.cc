#include "../inc/mempool.h"

#include <stdexcept>
#include <iostream>

namespace tensor {


template <>
Mempool<CPU>* Mempool<CPU>::instance_ = NULL;

template <>
Mempool<GPU>* Mempool<GPU>::instance_ = NULL;

// Mempool For CPU

template <>
Mempool<CPU>::Mempool () {
	re_malloc_time_ = 0;
}

template <>
Mempool<CPU>::~Mempool () {
	for (auto i = index_.begin(); i != index_.end(); ++i) {
		void* temp = i->first;
		Mem<CPU>::free(temp);
	}

	if (idle_.size() != index_.size()) {
		std::cerr<<"idle size: "<<idle_.size()<<"\n";
		std::cerr<<"index size: "<<index_.size()<<"\n";
		throw std::logic_error("~Mempool<CPU> : different idle and index size");
	}

	// std::cerr<<"CPU malloc: "<<index_.size()<<"\t";
	// std::cerr<<"re-malloc: "<<re_malloc_time_<<"\n";
}


template <> template <typename T>
T* Mempool<CPU>::malloc (int size) {
	int bytes = size * sizeof(T);
  auto p = std::make_pair(bytes, 1);
  auto i = idle_.find(p); // try find an idle one

  if (i == idle_.end()) {
    void* temp = NULL;
		try {
			temp = Mem<CPU>::malloc(bytes);
		} catch (std::exception& e) {}
		if (temp == NULL) {
			this->free_idle();
			temp = Mem<CPU>::malloc(bytes);
	    index_.insert(std::make_pair(temp, std::make_tuple(bytes, 1, 0)));
		} else {
			index_.insert(std::make_pair(temp, std::make_tuple(bytes, 1, 0)));
		}
    return (T*)temp;
  } else {
    re_malloc_time_++;
    void* temp = i->second.first;
    idle_.erase(i);
    return (T*)temp;
  }
}

template float* Mempool<CPU>::malloc<float> (int);
template double* Mempool<CPU>::malloc<double> (int);


template <> template <typename T>
void Mempool<CPU>::free (T* ptr) {
  auto i = index_.find(ptr);

  if (i == index_.end()) {
		std::cerr<<ptr<<'\t'<<idle_.size()<<'\t'<<index_.size()<<'\n';
		throw std::logic_error("Mempool<CPU>: bad free, no malloc");
  } else {
    idle_.insert( std::make_pair( std::make_pair(std::get<0>(i->second), std::get<1>(i->second)),
      std::make_pair(ptr, std::get<2>(i->second)) ) );
  }

	if (idle_.size() == index_.size()) {
		delete Mempool<CPU>::instance_;
		Mempool<CPU>::instance_ = NULL;
	}
}

template void Mempool<CPU>::free<float> (float*);
template void Mempool<CPU>::free<double> (double*);



template <>
void Mempool<CPU>::free_idle () {
	for (auto i = idle_.begin(); i != idle_.end(); ++i) {
		void* temp = i->second.first;
		auto j = index_.find(temp);
		if (j == index_.end()) {
			throw std::logic_error("Mempool<CPU>: bad free idle, no malloc");
		}
		index_.erase(j);
		Mem<CPU>::free(temp);
	}
	idle_.clear();
}

template void Mempool<CPU>::free_idle ();






// Mempool For GPU

template <>
Mempool<GPU>::Mempool () {
	re_malloc_time_ = 0;
}

template <>
Mempool<GPU>::~Mempool () {
	for (auto i = index_.begin(); i != index_.end(); ++i) {
		void* temp = i->first;
		Mem<GPU>::free(temp);
	}

	if (idle_.size() != index_.size()) {
		std::cerr<<"idle size: "<<idle_.size()<<"\n";
		std::cerr<<"index size: "<<index_.size()<<"\n";
		throw std::logic_error("~Mempool<GPU> : different idle and index size");
	}

	// std::cerr<<"CPU malloc: "<<index_.size()<<"\t";
	// std::cerr<<"re-malloc: "<<re_malloc_time_<<"\n";
}


template <> template <typename T>
T* Mempool<GPU>::malloc (int size) {
	int bytes = size * sizeof(T);
  auto p = std::make_pair(bytes, 1);
  auto i = idle_.find(p); // try find an idle one

  if (i == idle_.end()) {
    void* temp = NULL;
		try {
			temp = Mem<GPU>::malloc(bytes);
		} catch (std::exception& e) {}
		if (temp == NULL) {
			this->free_idle();
			temp = Mem<GPU>::malloc(bytes);
	    index_.insert(std::make_pair(temp, std::make_tuple(bytes, 1, 0)));
		} else {
			index_.insert(std::make_pair(temp, std::make_tuple(bytes, 1, 0)));
		}
    return (T*)temp;
  } else {
    re_malloc_time_++;
    void* temp = i->second.first;
    idle_.erase(i);
    return (T*)temp;
  }
}

template float* Mempool<GPU>::malloc<float> (int);
template double* Mempool<GPU>::malloc<double> (int);

template <> template <typename T>
void Mempool<GPU>::free (T* ptr) {
  auto i = index_.find(ptr);

  if (i == index_.end()) {
		std::cerr<<ptr<<'\t'<<idle_.size()<<'\t'<<index_.size()<<'\n';
		throw std::logic_error("Mempool<CPU>: bad free, no malloc");
  } else {
    idle_.insert( std::make_pair( std::make_pair(std::get<0>(i->second), std::get<1>(i->second)),
      std::make_pair(ptr, std::get<2>(i->second)) ) );
  }

	if (idle_.size() == index_.size()) {
		delete Mempool<GPU>::instance_;
		Mempool<GPU>::instance_ = NULL;
	}

}

template void Mempool<GPU>::free<float> (float*);
template void Mempool<GPU>::free<double> (double*);


template <>
void Mempool<GPU>::free_idle () {
	for (auto i = idle_.begin(); i != idle_.end(); ++i) {
		void* temp = i->second.first;
		auto j = index_.find(temp);
		if (j == index_.end()) {
			throw std::logic_error("Mempool<GPU>: bad free idle, no malloc");
		}
		index_.erase(j);
		Mem<GPU>::free(temp);
	}
	idle_.clear();
}

template void Mempool<GPU>::free_idle ();


}
