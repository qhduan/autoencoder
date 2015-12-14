
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

BOOST_AUTO_TEST_CASE( mat_construction )
{
  BOOST_CHECK( mat(std::vector<float>({1, 2, 3})) == mat("1; 2; 3") );
  BOOST_CHECK( mat({1, 2, 3}) == mat("1; 2; 3") );

  BOOST_CHECK( mat("1 2 3 4 5 6 7 8 9").reshape(3, 3) == mat("1 4 7; 2 5 8; 3 6 9") );

  BOOST_CHECK( mat("1 2 3; 4 5 6; 7 8 9") == mat("1 2 3; 4 5 6; 7 8 9;") );

  BOOST_CHECK( mat("1 2 3; 4 5 6; 7 8 9") == mat("1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0;") );

  BOOST_CHECK( mat("1 2 3; 4 5 6; 7 8 9") == mat("1 2 3; 4 5 6;   7 8 9;") );

  BOOST_CHECK( mat("1 2 3; 4 5 6; 7 8 9") == mat(" 1   2  3  ;  4 5 6  ;  7 8 9  ;  ") );

  BOOST_CHECK( mat(row({1, 2, 3})) == mat("1 2 3") );

  BOOST_CHECK( mat(col({1, 2, 3})) == mat("1; 2; 3") );

  BOOST_CHECK( mat(3, 3, tensor::fill::zeros) == mat("0 0 0; 0 0 0; 0 0 0") );

  BOOST_CHECK( mat(3, 3, tensor::fill::ones) == mat("1 1 1; 1 1 1; 1 1 1") );

  BOOST_CHECK( mat(3, 3, tensor::fill::eye) == mat("1 0 0; 0 1 0; 0 0 1") );
}

BOOST_AUTO_TEST_CASE( mat_submat )
{

  mat a("1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16");
  BOOST_CHECK( a.submat(0, 1, 0, 1) == mat("1 2; 5 6") );
  BOOST_CHECK( a.submat(1, 2, 1, 2) == mat("6 7; 10 11") );
  BOOST_CHECK( a.submat(2, 3, 2, 3) == mat("11 12; 15 16") );

  BOOST_CHECK( a.submat(0, 2, 0, 2) == mat("1 2 3; 5 6 7; 9 10 11") );
  BOOST_CHECK( a.submat(1, 3, 1, 3) == mat("6 7 8; 10 11 12; 14 15 16") );

  BOOST_CHECK( a.submat(0, 3, 0, 3) == mat("1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16") );

  for (int i = 0; i < a.n_rows; i++) {
    for (int j = 0; j < a.n_cols; j++) {
      a.submat(i, i, j, j) == a.submat(i, i, j, j);
    }
  }

  for (int i = 0; i < a.n_rows; i++) {
    for (int j = 0; j < a.n_cols; j++) {
      a.submat(i, (a.n_rows-1), j, (a.n_cols-1)) == a.submat(i, (a.n_rows-1), j, (a.n_cols-1));
    }
  }

  BOOST_CHECK( a.submat(0, 1, 0, 1) == a.submat(0, 1, 0, 1) );
  BOOST_CHECK( a.submat(1, 2, 1, 2) == a.submat(1, 2, 1, 2) );
  BOOST_CHECK( a.submat(2, 3, 2, 3) == a.submat(2, 3, 2, 3) );
  BOOST_CHECK( a.submat(0, 3, 0, 3) == a.submat(0, 3, 0, 3) );

  BOOST_CHECK( a.head_rows(1) == mat("1 2 3 4") );
  BOOST_CHECK( a.head_rows(2) == mat("1 2 3 4; 5 6 7 8") );
  BOOST_CHECK( a.head_rows(3) == mat("1 2 3 4; 5 6 7 8; 9 10 11 12") );
  BOOST_CHECK( a.head_rows(4) == mat("1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16") );

  BOOST_CHECK( a.tail_rows(1) == mat("13 14 15 16") );
  BOOST_CHECK( a.tail_rows(2) == mat("9 10 11 12; 13 14 15 16") );
  BOOST_CHECK( a.tail_rows(3) == mat("5 6 7 8; 9 10 11 12; 13 14 15 16") );
  BOOST_CHECK( a.tail_rows(4) == mat("1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16") );

  BOOST_CHECK( a.head_cols(1) == mat("1 ; 5 ; 9; 13") );
  BOOST_CHECK( a.head_cols(2) == mat("1 2 ; 5 6 ; 9 10; 13 14") );
  BOOST_CHECK( a.head_cols(3) == mat("1 2 3 ; 5 6 7 ; 9 10 11; 13 14 15") );
  BOOST_CHECK( a.head_cols(4) == mat("1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16") );

  BOOST_CHECK( a.tail_cols(1) == mat("4 ; 8; 12; 16") );
  BOOST_CHECK( a.tail_cols(2) == mat("3 4; 7 8; 11 12; 15 16") );
  BOOST_CHECK( a.tail_cols(3) == mat("2 3 4; 6 7 8; 10 11 12; 14 15 16") );
  BOOST_CHECK( a.tail_cols(4) == mat("1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16") );

  const mat& cref = a;
  BOOST_CHECK( cref.submat(0, 0, 0, 0) == a.submat(0, 0, 0, 0) );
  BOOST_CHECK( cref.submat(0, 1, 0, 1) == a.submat(0, 1, 0, 1) );
  BOOST_CHECK( cref.submat(0, 2, 0, 2) == a.submat(0, 2, 0, 2) );
  BOOST_CHECK( cref.submat(0, 3, 0, 3) == a.submat(0, 3, 0, 3) );

  BOOST_CHECK( cref.submat(1, 1, 1, 1) == a.submat(1, 1, 1, 1) );
  BOOST_CHECK( cref.submat(1, 2, 1, 2) == a.submat(1, 2, 1, 2) );
  BOOST_CHECK( cref.submat(1, 3, 1, 3) == a.submat(1, 3, 1, 3) );

  BOOST_CHECK( cref.submat(2, 2, 2, 2) == a.submat(2, 2, 2, 2) );
  BOOST_CHECK( cref.submat(2, 3, 2, 3) == a.submat(2, 3, 2, 3) );

  BOOST_CHECK( cref.submat(3, 3, 3, 3) == a.submat(3, 3, 3, 3) );
}


/*
 * Mat operation test
 */

BOOST_AUTO_TEST_CASE( mat_operation_1 )
{

  // mat += value

  BOOST_CHECK( (mat("1 2 3; 4 5 6") += 2.0) == mat("3 4 5; 6 7 8") );
  BOOST_CHECK( (mat("1 2; 3 4; 5 6") += 2.0) == mat("3 4; 5 6; 7 8") );

  BOOST_CHECK( (mat("1 2 3; 4 5 6") -= 2.0) == mat("-1 0 1; 2 3 4") );
  BOOST_CHECK( (mat("1 2; 3 4; 5 6") -= 2.0) == mat("-1 0; 1 2; 3 4") );

  BOOST_CHECK( (mat("1 2 3; 4 5 6") *= 2.0) == mat("2 4 6; 8 10 12") );
  BOOST_CHECK( (mat("1 2; 3 4; 5 6") *= 2.0) == mat("2 4; 6 8; 10 12") );

  BOOST_CHECK( (mat("1 2 3; 4 5 6") /= 0.5) == mat("2 4 6; 8 10 12") );
  BOOST_CHECK( (mat("1 2; 3 4; 5 6") /= 0.5) == mat("2 4; 6 8; 10 12") );

  // mat + value

  BOOST_CHECK( (mat("1 2 3; 4 5 6") + 2.0) == mat("3 4 5; 6 7 8") );
  BOOST_CHECK( (mat("1 2; 3 4; 5 6") + 2.0) == mat("3 4; 5 6; 7 8") );

  BOOST_CHECK( (mat("1 2 3; 4 5 6") - 2.0) == mat("-1 0 1; 2 3 4") );
  BOOST_CHECK( (mat("1 2; 3 4; 5 6") - 2.0) == mat("-1 0; 1 2; 3 4") );

  BOOST_CHECK( (mat("1 2 3; 4 5 6") * 2.0) == mat("2 4 6; 8 10 12") );
  BOOST_CHECK( (mat("1 2; 3 4; 5 6") * 2.0) == mat("2 4; 6 8; 10 12") );

  BOOST_CHECK( (mat("1 2 3; 4 5 6") / 0.5) == mat("2 4 6; 8 10 12") );
  BOOST_CHECK( (mat("1 2; 3 4; 5 6") / 0.5) == mat("2 4; 6 8; 10 12") );

  // mat += mat

  BOOST_CHECK( (mat("1 2 3; 4 5 6") += mat("2 2 2; 2 2 2")) == mat("3 4 5; 6 7 8") );
  BOOST_CHECK( (mat("1 2; 3 4; 5 6") += mat("2 2; 2 2; 2 2")) == mat("3 4; 5 6; 7 8") );

  BOOST_CHECK( (mat("1 2 3; 4 5 6") -= mat("2 2 2; 2 2 2")) == mat("-1 0 1; 2 3 4") );
  BOOST_CHECK( (mat("1 2; 3 4; 5 6") -= mat("2 2; 2 2; 2 2")) == mat("-1 0; 1 2; 3 4") );

  BOOST_CHECK( (mat("1 2 3; 4 5 6") %= mat("2 2 2; 2 2 2")) == mat("2 4 6; 8 10 12") );
  BOOST_CHECK( (mat("1 2; 3 4; 5 6") %= mat("2 2; 2 2; 2 2")) == mat("2 4; 6 8; 10 12") );

  BOOST_CHECK( (mat("1 2 3; 4 5 6") /= mat("0.5 0.5 0.5; 0.5 0.5 0.5")) == mat("2 4 6; 8 10 12") );
  BOOST_CHECK( (mat("1 2; 3 4; 5 6") /= mat("0.5 0.5; 0.5 0.5; 0.5 0.5")) == mat("2 4; 6 8; 10 12") );

  // mat + mat

  BOOST_CHECK( (mat("1 2 3; 4 5 6") + mat("2 2 2; 2 2 2")) == mat("3 4 5; 6 7 8") );
  BOOST_CHECK( (mat("1 2; 3 4; 5 6") + mat("2 2; 2 2; 2 2")) == mat("3 4; 5 6; 7 8") );

  BOOST_CHECK( (mat("1 2 3; 4 5 6") - mat("2 2 2; 2 2 2")) == mat("-1 0 1; 2 3 4") );
  BOOST_CHECK( (mat("1 2; 3 4; 5 6") - mat("2 2; 2 2; 2 2")) == mat("-1 0; 1 2; 3 4") );

  BOOST_CHECK( (mat("1 2 3; 4 5 6") % mat("2 2 2; 2 2 2")) == mat("2 4 6; 8 10 12") );
  BOOST_CHECK( (mat("1 2; 3 4; 5 6") % mat("2 2; 2 2; 2 2")) == mat("2 4; 6 8; 10 12") );

  BOOST_CHECK( (mat("1 2 3; 4 5 6") / mat("0.5 0.5 0.5; 0.5 0.5 0.5")) == mat("2 4 6; 8 10 12") );
  BOOST_CHECK( (mat("1 2; 3 4; 5 6") / mat("0.5 0.5; 0.5 0.5; 0.5 0.5")) == mat("2 4; 6 8; 10 12") );
}

BOOST_AUTO_TEST_CASE( mat_operation_2 )
{
  mat a, b;

  a = "1 2; 3 4";

  BOOST_CHECK( (a*a*a*a*a) == mat(" 1069 1558; 2337 3406 ") );

  a = "1 2 3";
  b = "4; 5; 6";

  BOOST_CHECK( (a*b) == mat(" 32 ") );
  BOOST_CHECK( (b*a) == mat(" 4 8 12; 5 10 15; 6 12 18 ") );
}

/*
 * Mat function test
 */

BOOST_AUTO_TEST_CASE( mat_function_exp )
{
  BOOST_CHECK( mat(4, 2).fill(0).exp() == mat(4, 2).fill(1) );
  BOOST_CHECK( mat(2, 4).fill(0).exp() == mat(2, 4).fill(1) );
  BOOST_CHECK( mat(4, 2).fill(-1000000).exp() == mat(4, 2).fill(0) );
  BOOST_CHECK( mat(2, 4).fill(-1000000).exp() == mat(2, 4).fill(0) );
}

BOOST_AUTO_TEST_CASE( mat_function_log )
{
  BOOST_CHECK( mat(4, 2).fill(1).log() == mat(4, 2).fill(0) );
  BOOST_CHECK( mat(2, 4).fill(1).log() == mat(2, 4).fill(0) );
  BOOST_CHECK( mat(4, 2).fill(148.4131591025766).log() == mat(4, 2).fill(5) );
  BOOST_CHECK( mat(2, 4).fill(148.4131591025766).log() == mat(2, 4).fill(5) );
}

BOOST_AUTO_TEST_CASE( mat_function_abs )
{
  BOOST_CHECK( mat(4, 2).fill(-1).abs() == mat(4, 2).fill(1) );
  BOOST_CHECK( mat(2, 4).fill(-1).abs() == mat(2, 4).fill(1) );
  BOOST_CHECK( mat(4, 2).fill(1).abs() == mat(4, 2).fill(1) );
  BOOST_CHECK( mat(2, 4).fill(1).abs() == mat(2, 4).fill(1) );
}

BOOST_AUTO_TEST_CASE( mat_function_sigmoid )
{
  BOOST_CHECK( mat(10, 10).fill(0).sigmoid() == mat(10, 10).fill(0.5) );
  BOOST_CHECK( mat(10, 10).fill(1000).sigmoid() == mat(10, 10).fill(1) );
  BOOST_CHECK( mat(10, 10).fill(-1000).sigmoid() == mat(10, 10).fill(0) );
}

BOOST_AUTO_TEST_CASE( mat_function_dsigmoid )
{
  BOOST_CHECK( mat(10, 10).fill(0).dsigmoid() == mat(10, 10).fill(0.25) );
  BOOST_CHECK( mat(10, 10).fill(2).dsigmoid() == mat(10, 10).fill(0.1049935854035065) );
}

BOOST_AUTO_TEST_CASE( mat_function_tanh )
{
  BOOST_CHECK( mat(10, 10).fill(0).tanh() == mat(10, 10).fill(0) );
  BOOST_CHECK( mat(10, 10).fill(1000).tanh() == mat(10, 10).fill(1) );
  BOOST_CHECK( mat(10, 10).fill(-1000).tanh() == mat(10, 10).fill(-1) );
}

BOOST_AUTO_TEST_CASE( mat_function_pow )
{
  BOOST_CHECK( mat(10, 10).fill(2).pow(2) == mat(10, 10).fill(4) );
  BOOST_CHECK( mat(10, 10).fill(3).pow(3) == mat(10, 10).fill(27) );
  BOOST_CHECK( mat(10, 10).fill(-2).pow(2) == mat(10, 10).fill(4) );
  BOOST_CHECK( mat(10, 10).fill(-3).pow(3) == mat(10, 10).fill(-27) );
}

BOOST_AUTO_TEST_CASE( mat_function_sqrt )
{
  BOOST_CHECK( mat(10, 10).fill(4).sqrt() == mat(10, 10).fill(2) );
  BOOST_CHECK( mat(10, 10).fill(9).sqrt() == mat(10, 10).fill(3) );
}

BOOST_AUTO_TEST_CASE( mat_function_sign )
{
  BOOST_CHECK( mat(10, 10).fill(1.5).sign() == mat(10, 10).fill(1) );
  BOOST_CHECK( mat(10, 10).fill(-1.5).sign() == mat(10, 10).fill(-1) );
  BOOST_CHECK( mat(10, 10).fill(0).sign() == mat(10, 10).fill(0) );
}

BOOST_AUTO_TEST_CASE( mat_function_floor )
{
  BOOST_CHECK( mat(10, 10).fill(1.5).floor() == mat(10, 10).fill(1) );
  BOOST_CHECK( mat(10, 10).fill(-1.5).floor() == mat(10, 10).fill(-2) );
}

BOOST_AUTO_TEST_CASE( mat_function_ceil )
{
  BOOST_CHECK( mat(10, 10).fill(1.5).ceil() == mat(10, 10).fill(2) );
  BOOST_CHECK( mat(10, 10).fill(-1.5).ceil() == mat(10, 10).fill(-1) );
}

BOOST_AUTO_TEST_CASE( mat_function_round )
{
  BOOST_CHECK( mat(10, 10).fill(1.5).round() == mat(10, 10).fill(2) );
  BOOST_CHECK( mat(10, 10).fill(1.4).round() == mat(10, 10).fill(1) );
}

BOOST_AUTO_TEST_CASE( mat_function_swap )
{
  mat a(3, 3), b(3, 3);
  a.randu();
  b.randu();
  mat c = a;
  mat d = b;
  c.swap(d);
  BOOST_CHECK( c == b );
  BOOST_CHECK( d == a );
}

BOOST_AUTO_TEST_CASE( mat_function_swap_rows )
{
  mat a("1 2 3; 4 5 6; 7 8 9");
  a.swap_rows(0, 2);
  BOOST_CHECK( a == mat("7 8 9; 4 5 6; 1 2 3") );
}

BOOST_AUTO_TEST_CASE( mat_function_swap_cols )
{
  mat a("1 2 3; 4 5 6; 7 8 9");
  a.swap_cols(0, 2);
  BOOST_CHECK( a == mat("3 2 1; 6 5 4; 9 8 7") );
}

BOOST_AUTO_TEST_CASE( mat_function_each_col )
{
  mat a;
  mat b("1; 2; 3");
  col c({1, 2, 3});

  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_col() += b;
  BOOST_CHECK( a == mat("2 3 4; 6 7 8; 10 11 12") );
  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_col() += c;
  BOOST_CHECK( a == mat("2 3 4; 6 7 8; 10 11 12") );

  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_col() -= b;
  BOOST_CHECK( a == mat("0 1 2; 2 3 4; 4 5 6") );
  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_col() -= c;
  BOOST_CHECK( a == mat("0 1 2; 2 3 4; 4 5 6") );

  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_col() %= b;
  BOOST_CHECK( a == mat("1 2 3; 8 10 12; 21 24 27") );
  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_col() %= c;
  BOOST_CHECK( a == mat("1 2 3; 8 10 12; 21 24 27") );

  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_col() /= b;
  BOOST_CHECK( a == mat("1 2 3; 2 2.5 3; 2.333333333333 2.666666666666 3") );

  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_col() /= c;
  BOOST_CHECK( a == mat("1 2 3; 2 2.5 3; 2.333333333333 2.666666666666 3") );

  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_col() = b;
  BOOST_CHECK( a == mat("1 1 1; 2 2 2; 3 3 3") );

  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_col() = c;
  BOOST_CHECK( a == mat("1 1 1; 2 2 2; 3 3 3") );
}

BOOST_AUTO_TEST_CASE( mat_function_each_row )
{
  mat a;
  mat b("1 2 3");
  b.reshape(1, 3);
  row c({1, 2, 3});

  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_row() += b;
  BOOST_CHECK( a == mat("2 4 6; 5 7 9; 8 10 12") );
  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_row() += c;
  BOOST_CHECK( a == mat("2 4 6; 5 7 9; 8 10 12") );

  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_row() -= b;
  BOOST_CHECK( a == mat("0 0 0; 3 3 3; 6 6 6") );
  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_row() -= c;
  BOOST_CHECK( a == mat("0 0 0; 3 3 3; 6 6 6") );

  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_row() %= b;
  BOOST_CHECK( a == mat("1 4 9; 4 10 18; 7 16 27") );
  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_row() %= c;
  BOOST_CHECK( a == mat("1 4 9; 4 10 18; 7 16 27") );

  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_row() /= b;
  BOOST_CHECK( a == mat("1 1 1; 4 2.5 2; 7 4 3") );

  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_row() /= c;
  BOOST_CHECK( a == mat("1 1 1; 4 2.5 2; 7 4 3") );

  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_row() = b;
  BOOST_CHECK( a == mat("1 2 3; 1 2 3; 1 2 3") );

  a = "1 2 3; 4 5 6; 7 8 9";
  a.each_row() = c;
  BOOST_CHECK( a == mat("1 2 3; 1 2 3; 1 2 3") );
}

BOOST_AUTO_TEST_CASE( mat_function_accu ) {
  mat a = tensor::ones<mat>(100, 100);
  BOOST_CHECK( accu(a) == 10000 );
  mat b = a * -1.0f;
  BOOST_CHECK( accu(b) == -10000 );
}

BOOST_AUTO_TEST_CASE( mat_function_norm ) {
  mat a("1 2 2");
  BOOST_CHECK( norm(a) == 3.0 );
  mat b = a * -1.0f;
  BOOST_CHECK( norm(b) == 3.0 );
}

BOOST_AUTO_TEST_CASE( mat_function_SumOfSquared ) {

  BOOST_CHECK( SumOfSquared(mat("1 2 3; 4 5 6")) == 91 );
  BOOST_CHECK( SumOfSquared(mat("1 2; 3 4; 5 6")) == 91 );

  BOOST_CHECK( SumOfSquared(mat("1 2 3; 4 5 6")) == accu(mat("1 2 3; 4 5 6").pow(2)) );
  BOOST_CHECK( SumOfSquared(mat("1 2; 3 4; 5 6")) == accu(mat("1 2; 3 4; 5 6").pow(2)) );

}

BOOST_AUTO_TEST_CASE( mat_function_transpose ) {
  mat a("1 2 3; 4 5 6; 7 8 9");
  mat b("1 2 3; 4 5 6; 7 8 9");
  BOOST_CHECK( a.t() == mat("1 4 7; 2 5 8; 3 6 9") );
  BOOST_CHECK( a == b );
  a = "1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16";
  BOOST_CHECK( a.t() == mat("1 5 9 13; 2 6 10 14; 3 7 11 15; 4 8 12 16") );
  a = "1 2; 3 4; 5 6; 7 8";
  BOOST_CHECK( a.t() == mat("1 3 5 7; 2 4 6 8") );

  a = "1 2; 3 4; 5 6";
  b = "2 3; 4 5; 6 7";
  BOOST_CHECK( (a.t() * b) == mat("44 53; 56 68") );

  BOOST_CHECK( (b * a.t()) == mat("8 18 28; 14 32 50; 20 46 72") );
}

BOOST_AUTO_TEST_CASE( mat_function_sum ) {
  BOOST_CHECK( sum(mat("1 2 3; 4 5 6; 7 8 9"), 0) == mat("12 15 18") );
  BOOST_CHECK( sum(mat("1 2 3; 4 5 6; 7 8 9"), 1) == mat("6; 15; 24") );
  BOOST_CHECK( sum(mat("1 2 3 4; 5 6 7 8"), 0) == mat("6 8 10 12") );
  BOOST_CHECK( sum(mat("1 2 3 4; 5 6 7 8"), 1) == mat("10; 26") );
  BOOST_CHECK( sum(mat("1 2; 3 4; 5 6; 7 8"), 0) == mat("16 20") );
  BOOST_CHECK( sum(mat("1 2; 3 4; 5 6; 7 8"), 1) == mat("3; 7; 11; 15") );
}

BOOST_AUTO_TEST_CASE( mat_function_submat ) {
  mat a("1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16");
  mat b("99 ; 99");
  a.submat(0, 1, 0, 0) = b;
  BOOST_CHECK( a == mat("99 2 3 4; 99 6 7 8; 9 10 11 12; 13 14 15 16") );
  a.submat(0, 1, 1, 1) = b;
  BOOST_CHECK( a == mat("99 99 3 4; 99 99 7 8; 9 10 11 12; 13 14 15 16") );
}

BOOST_AUTO_TEST_CASE( mat_function_min_max ) {
  mat a("1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16");

  BOOST_CHECK( max(a) == 16 );
  BOOST_CHECK( min(a) == 1 );

  BOOST_CHECK( max(a, 0) == mat("13 14 15 16") );
  BOOST_CHECK( max(a, 1) == mat("4; 8; 12; 16") );

  BOOST_CHECK( min(a, 0) == mat("1 2 3 4") );
  BOOST_CHECK( min(a, 1) == mat("1; 5; 9; 13") );

  mat b = a * -1.0f;

  BOOST_CHECK( max(b) == -1 );
  BOOST_CHECK( min(b) == -16 );
}
