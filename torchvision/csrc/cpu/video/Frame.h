#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <torch/custom_class.h>
#include <torch/script.h>

// typedef std::tuple<uint32_t, uint32_t, std::vector<string>> SubsData;

struct Frame : torch::CustomClassHolder {
  std::string type_;
  double pts_;
  torch::Tensor data_;
  //   SubsData subs_;

 public:
  Frame(std::string type, double pts, torch::Tensor data) {
    type_ = type;
    pts_ = pts;
    data_ = data;
  }
  //   explicit Frame(std::string type, std::double pts, SubsData data) {
  //     type_ = type;
  //     pts_ = pts;
  //     subs_ = data;
  //   }

  std::string getType() {
    return type_;
  }

  double getPTS() {
    return pts_;
  }

  torch::Tensor getData() {
    return data_;
  }
  //   SubsData getSubsData() {
  //     return subs_;
  //   }
}; // struct Frame
