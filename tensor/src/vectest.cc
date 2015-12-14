
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE MatTest
#include <boost/test/unit_test.hpp>

#include "../inc/tensor.hpp"

#ifdef MATTEST_GPU

typedef tensor::Mat<tensor::GPU, float> mat;
typedef tensor::Row<tensor::GPU, float> row;
typedef tensor::Col<tensor::GPU, float> col;

#else

typedef tensor::Mat<tensor::CPU, float> mat;
typedef tensor::Row<tensor::CPU, float> row;
typedef tensor::Col<tensor::CPU, float> col;

#endif

BOOST_AUTO_TEST_CASE( hello )
{

#ifdef MATTEST_GPU
std::cout << "Matrix Test GPU" << std::endl;
#else
std::cout << "Matrix Test CPU" << std::endl;
#endif

}

BOOST_AUTO_TEST_CASE( vec_construction_1 )
{
  row r("1 2 3 4");
  col c("1 2 3 4");
  r.print("row");
  c.print("col");
  //BOOST_CHECK( mat({1, 2, 3}) == mat(std::vector<float>({1, 2, 3})) );
}
