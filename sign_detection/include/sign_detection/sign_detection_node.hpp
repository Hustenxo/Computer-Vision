#ifndef SIGN_DETECTION_SIGN_DETECTION_NODE_HPP
#define SIGN_DETECTION_SIGN_DETECTION_NODE_HPP

#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"
#include "libpsaf/interface/sign_detection_interface.hpp"
#include "libpsaf_msgs/msg/sign.hpp"
#include "std_msgs/msg/float64.hpp"
#include "torch/script.h"
#include "torch/torch.h"
#include "sign_detection/definitions.hpp"

class SignDetectionNode : public libpsaf::SignDetectionInterface
{
public:
  /**
   * @brief Default constructor
   */
  SignDetectionNode();

  /**
   * @brief overloaded constructor with node name
   * @param name node name
   */
  explicit SignDetectionNode(std::string name);

protected:
  /**
   * @brief callback for images
   * @param img open cv matrix with image
   * @param sensor number of sensors
   */
  void processImage(cv::Mat & img, int sensor) final;

  /**
  * @brief callback function for speed
  * @param p
  */
  void updateSpeed(std_msgs::msg::Float64::SharedPtr p);

  /**
   * @brief create and publish a sign message
   * @param sign_detect Detection struct of detected sign
   * @param time time it took for the processing in milliseconds
   * @param speed speed of the car at the beginning of processing
   */
  void publishSignMessage(const Detection & sign_detect, double time, double speed);

  /**
   * @brief map the sign type to the numbers wanted in the state machine
   * @param detection index of detected sign in network
   * @return index of detected sign for state machine
   */
  static int mapDetToSignType(int detection);

  /**
   * @brief calculate the area of the bounding box
   * @param height height of bounding box
   * @param width width of bounding box
   * @return area
   */
  static double getBoundingBoxArea(int height, int width);

  /**
   * @brief determine distance from sign in y direction given the area of the bounding box
   * @param area area of the bounding box
   * @return distance in y direction in mm
   */
  static double getSignDistance(double area);

  /**
   * @brief resize a given image to a wanted size
   * @param src source image
   * @param dst destination image
   * @param out_size desired output size
   * @return padding information
   */
  static std::vector<float> getLetterboxImage(
    const cv::Mat & src, cv::Mat & dst, const
    cv::Size & out_size);

  /**
   * @brief draw detected bounding boxes into the image and publish it
   * @param img image
   * @param detections detected signs
   * @param class_names file with all class names
   * @param label true if label should be printed
   */
  void publishImageWithBoundingBox(
    cv::Mat & img,
    const std::vector<Detection> & detections,
    const std::vector<std::string> & class_names,
    bool label = true);

  /**
   * @brief preprocess the output of the neural network
   * @param detections detection tensor
   * @param pad_w amount of padding in width direction
   * @param pad_h amount of padding in height direction
   * @param scale factor with which image was scaled
   * @param img_shape original image shape
   * @param conf_thres confidence threshold
   * @param iou_thres threshold for IoU
   * @return vector with all detections
   */
  static std::vector<Detection> postProcessing(
    const torch::Tensor & detections,
    float pad_w, float pad_h, float scale, const cv::Size & img_shape,
    float conf_thres = 0.4, float iou_thres = 0.5);

  /**
   * @brief scales the detected bounding boxes to fit to the original image size
   * @param data detections
   * @param pad_w amount of padding in width direction
   * @param pad_h amount of padding in height direction
   * @param scale factor with which image was scaled
   * @param img_shape img_shape original image shape
   */
  static void scaleCoordinates(
    std::vector<Detection> & data, float pad_w, float pad_h,
    float scale, const cv::Size & img_shape);

  /**
   * @brief convert bounding box format
   * @param x tensor with indices
   * @return new indices
   */
  static torch::Tensor xywh2xyxy(const torch::Tensor & x);

  /**
   * @brief convert detections from tensor to Detection struct
   * @param offset_boxes offset of the boxes
   * @param det detections
   * @param offset_box_vec offset box vector
   * @param score_vec score vector
   */
  static void tensor2Detection(
    const at::TensorAccessor<float, 2> & offset_boxes,
    const at::TensorAccessor<float, 2> & det,
    std::vector<cv::Rect> & offset_box_vec,
    std::vector<float> & score_vec);

  /**
   * @brief callback for state, not implemented
   * @param prevState previous state
   * @param newState new state
   */
  void onStateChange(int prevState, int newState) final;

  // variable for speed
  double current_speed_;

private:
  std::string node_name;

  // module and class names
  torch::jit::script::Module module_;
  std::vector<std::string> class_names_;

  // variables for different thresholds
  float conf_threshold_;
  float iou_threshold_;

  // subscribers
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr speed_subscriber_;
};

#endif  // SIGN_DETECTION_SIGN_DETECTION_NODE_HPP
