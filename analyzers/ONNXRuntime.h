/*
 * Based on CMSSW ONNXRuntime.h : https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h
 * A convinient C++ wrapper for ONNXRuntime C++ API.
 */

#ifndef ONNXRUNTIME_H_
#define ONNXRUNTIME_H_

#include <vector>
#include <map>
#include <string>
#include <memory>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <functional>

#include "onnxruntime_cxx_api.h"

namespace myOrt {

  typedef std::vector<std::vector<float>> FloatArrays;

  class ONNXRuntime {
  public:
    ONNXRuntime(const std::string& model_path, const ::Ort::SessionOptions* session_options = nullptr);
    ONNXRuntime(const ONNXRuntime&) = delete;
    ONNXRuntime& operator=(const ONNXRuntime&) = delete;
    ~ONNXRuntime();

    // Run inference and get outputs
    // input_names: list of the names of the input nodes.
    // input_values: list of input arrays for each input node. The order of `input_values` must match `input_names`.
    // input_shapes: list of `int64_t` arrays specifying the shape of each input node. Can leave empty if the model does not have dynamic axes.
    // output_names: names of the output nodes to get outputs from. Empty list means all output nodes.
    // batch_size: number of samples in the batch. Each array in `input_values` must have a shape layout of (batch_size, ...).
    // Returns: a std::vector<std::vector<float>>, with the order matched to `output_names`.
    // When `output_names` is empty, will return all outputs ordered as in `getOutputNames()`.
    FloatArrays run(const std::vector<std::string>& input_names,
                    FloatArrays& input_values,
                    const std::vector<std::vector<int64_t>>& input_shapes = {},
                    const std::vector<std::string>& output_names = {},
                    int64_t batch_size = 1) const;

    // Get a list of names of all the output nodes
    const std::vector<std::string>& getOutputNames() const;

    // Get the shape of a output node
    // The 0th dim depends on the batch size, therefore is set to -1
    const std::vector<int64_t>& getOutputShape(const std::string& output_name) const;

  private:
    static const ::Ort::Env& env() {
        static ::Ort::Env instance(ORT_LOGGING_LEVEL_ERROR, "");
        return instance;
    }
    std::unique_ptr<::Ort::Session> session_;

    std::vector<std::string> input_node_strings_;
    std::vector<const char*> input_node_names_;
    std::map<std::string, std::vector<int64_t>> input_node_dims_;

    std::vector<std::string> output_node_strings_;
    std::vector<const char*> output_node_names_;
    std::map<std::string, std::vector<int64_t>> output_node_dims_;
  };


  using namespace ::Ort;

  ONNXRuntime::ONNXRuntime(const std::string& model_path, const SessionOptions* session_options) {
    // create session
    if (session_options) {
      session_.reset(new Session(env(), model_path.c_str(), *session_options));
    } else {
      SessionOptions sess_opts;
      sess_opts.SetIntraOpNumThreads(1);
      session_.reset(new Session(env(), model_path.c_str(), sess_opts));
    }
    AllocatorWithDefaultOptions allocator;

    // get input names and shapes
    size_t num_input_nodes = session_->GetInputCount();
    input_node_strings_.resize(num_input_nodes);
    input_node_names_.resize(num_input_nodes);
    input_node_dims_.clear();

    for (size_t i = 0; i < num_input_nodes; i++) {
      // get input node names
      std::string input_name(&*session_->GetInputNameAllocated(i, allocator));
      input_node_strings_[i] = input_name;
      input_node_names_[i] = input_node_strings_[i].c_str();

      // get input shapes
      auto type_info = session_->GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      input_node_dims_[input_name] = tensor_info.GetShape();
    }

    size_t num_output_nodes = session_->GetOutputCount();
    output_node_strings_.resize(num_output_nodes);
    output_node_names_.resize(num_output_nodes);
    output_node_dims_.clear();

    for (size_t i = 0; i < num_output_nodes; i++) {
      // get output node names
      std::string output_name(&*session_->GetOutputNameAllocated(i, allocator));
      output_node_strings_[i] = output_name;
      output_node_names_[i] = output_node_strings_[i].c_str();

      // get output node types
      auto type_info = session_->GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      output_node_dims_[output_name] = tensor_info.GetShape();

      // the 0th dim depends on the batch size
      output_node_dims_[output_name].at(0) = -1;
    }
  }

  FloatArrays ONNXRuntime::run(const std::vector<std::string>& input_names,
                               FloatArrays& input_values,
                               const std::vector<std::vector<int64_t>>& input_shapes,
                               const std::vector<std::string>& output_names,
                               int64_t batch_size) const {
    assert(input_names.size() == input_values.size());
    assert(input_shapes.empty() || input_names.size() == input_shapes.size());
    assert(batch_size > 0);

    // create input tensor objects from data values
    std::vector<Value> input_tensors;
    auto memory_info = MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (const auto& name : input_node_strings_) {
      auto iter = std::find(input_names.begin(), input_names.end(), name);
      if (iter == input_names.end()) {
        throw std::runtime_error("Input " + name + " is not provided!");
      }
      auto input_pos = iter - input_names.begin();
      auto value = input_values.begin() + input_pos;
      std::vector<int64_t> input_dims;
      if (input_shapes.empty()) {
        input_dims = input_node_dims_.at(name);
        input_dims[0] = batch_size;
      } else {
        input_dims = input_shapes[input_pos];
        // rely on the given input_shapes to set the batch size
      }
      auto expected_len = std::accumulate(input_dims.begin(), input_dims.end(), 1, std::multiplies<int64_t>());
      if (expected_len != (int64_t)value->size()) {
        throw std::runtime_error(
            "Input array " + name + " has a wrong size of " + std::to_string(value->size()) + ", expected " + std::to_string(expected_len)
        );
      }
      auto input_tensor =
          Value::CreateTensor<float>(memory_info, value->data(), value->size(), input_dims.data(), input_dims.size());
      assert(input_tensor.IsTensor());
      input_tensors.emplace_back(std::move(input_tensor));
    }

    // set output node names; will get all outputs if `output_names` is not provided
    std::vector<const char*> run_output_node_names;
    if (output_names.empty()) {
      run_output_node_names = output_node_names_;
    } else {
      for (const auto& name : output_names) {
        run_output_node_names.push_back(name.c_str());
      }
    }

    // run
    auto output_tensors = session_->Run(RunOptions{nullptr},
                                        input_node_names_.data(),
                                        input_tensors.data(),
                                        input_tensors.size(),
                                        run_output_node_names.data(),
                                        run_output_node_names.size());

    // convert output to floats
    FloatArrays outputs;
    for (auto& output_tensor : output_tensors) {
      assert(output_tensor.IsTensor());

      // get output shape
      auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
      auto length = tensor_info.GetElementCount();

      auto floatarr = output_tensor.GetTensorMutableData<float>();
      outputs.emplace_back(floatarr, floatarr + length);
    }
    assert(outputs.size() == run_output_node_names.size());

    return outputs;
  }

  const std::vector<std::string>& ONNXRuntime::getOutputNames() const {
    if (session_) {
      return output_node_strings_;
    } else {
      throw std::runtime_error("Needs to call createSession() first before getting the output names!");
    }
  }

  const std::vector<int64_t>& ONNXRuntime::getOutputShape(const std::string& output_name) const {
    auto iter = output_node_dims_.find(output_name);
    if (iter == output_node_dims_.end()) {
      throw std::runtime_error("Output name " + output_name + " is invalid!");
    } else {
      return iter->second;
    }
  }

}

#endif
