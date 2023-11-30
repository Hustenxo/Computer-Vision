#ifndef SIGN_DETECTION__DEFINITIONS_HPP_
#define SIGN_DETECTION__DEFINITIONS_HPP_

// definitions for different signs
#define STOP 0u
#define LIMIT_30 1u
#define END_LIMIT_30 11u
#define PEDESTRIAN 12u
#define YIELD 13u

// different indices for detection
enum Det
{
  tl_x = 0,
  tl_y = 1,
  br_x = 2,
  br_y = 3,
  score = 4,
  class_idx = 5
};

// Detection struct
struct Detection
{
  cv::Rect bbox;
  float score;
  int class_idx;
};

#endif  // SIGN_DETECTION__DEFINITIONS_HPP_
