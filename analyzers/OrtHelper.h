#ifndef OrtHelper_h
#define OrtHelper_h

#include <algorithm>
#include <iostream>

#include "ONNXRuntime.h"

class OrtHelper {
public:
    OrtHelper(std::string model_path, bool debug = false) {
        ort_ = std::make_unique<myOrt::ONNXRuntime>(model_path);
        debug_ = debug;
        init_data();
    }
    ~OrtHelper() {}

    void infer_model(std::map<std::string, std::vector<float>>& particleVars, std::map<std::string, float>& jetVars) {

        // Extract input and perform preprocessing
        make_input(particleVars, jetVars);

        // Inference via onnxruntime
        output_ = ort_->run(input_names_, data_, input_shapes_)[0];
        if (debug_) {
            std::cout << "model output (size = " << output_.size() << "):\n";
            for (auto v: output_) {
                std::cout << v << " ";
            }
            std::cout << std::endl;
        }
    }

    std::vector<float>& get_output() {
        return output_;
    }

private:
    std::unique_ptr<myOrt::ONNXRuntime> ort_ = nullptr;
    std::vector<std::string> input_names_ = {"pf_features", "pf_vectors", "pf_mask"};
    std::vector<std::vector<int64_t>> input_shapes_ = {{1, 17, 128}, {1, 4, 128}, {1, 1, 128}}; // (batch_size=1, channel, length)
    std::vector<std::vector<std::tuple<std::string, float, float, float, float>>> input_var_info_ = {
        // (name, subtract_val, multiply_val, clip_min, clip_max)
        {
            {"part_pt_scale_log", 1.7, 0.7, -5, 5},
            {"part_e_scale_log", 2.0, 0.7, -5, 5},
            {"part_logptrel", -4.7, 0.7, -5, 5},
            {"part_logerel", -4.7, 0.7, -5, 5},
            {"part_deltaR", 0.2, 4.0, -5, 5},
            {"part_charge", 0, 1, -1e8, 1e8},
            {"part_isChargedHadron", 0, 1, -1e8, 1e8},
            {"part_isNeutralHadron", 0, 1, -1e8, 1e8},
            {"part_isPhoton", 0, 1, -1e8, 1e8},
            {"part_isElectron", 0, 1, -1e8, 1e8},
            {"part_isMuon", 0, 1, -1e8, 1e8},
            {"part_d0", 0, 1, -1e8, 1e8},
            {"part_d0err", 0, 1, 0, 1},
            {"part_dz", 0, 1, -1e8, 1e8},
            {"part_dzerr", 0, 1, 0, 1},
            {"part_deta", 0, 1, -1e8, 1e8},
            {"part_dphi", 0, 1, -1e8, 1e8}
        },
        {
            {"part_px_scale", 0, 1, -1e8, 1e8},
            {"part_py_scale", 0, 1, -1e8, 1e8},
            {"part_pz_scale", 0, 1, -1e8, 1e8},
            {"part_energy_scale", 0, 1, -1e8, 1e8}
        },
        {
            {"part_mask", 0, 1, -1e8, 1e8}
        }
    };
    std::map<std::string, std::vector<float>> input_feats_;
    std::vector<std::vector<float>> data_;
    std::vector<float> output_;
    bool debug_ = false;

    void init_data() {
        // initialize the data_ vector
        for (size_t i = 0; i < input_names_.size(); i++) {
            data_.emplace_back(input_shapes_[i][1] * input_shapes_[i][2], 0);
        }
        // initialize input_feats_
        for (auto v: std::vector<std::string>({"part_deta", "part_dphi", "part_charge", "part_d0err", "part_dzerr", "part_px_scale", "part_py_scale", "part_pz_scale", "part_energy_scale", "part_pt_scale", "part_pt_scale_log", "part_e_scale_log", "part_logptrel", "part_logerel", "part_deltaR", "part_d0", "part_dz", "part_isElectron", "part_isMuon", "part_isPhoton", "part_isChargedHadron", "part_isNeutralHadron", "part_mask"})) {
            input_feats_[v] = std::vector<float>();
        }
    }

    void make_input(std::map<std::string, std::vector<float>>& particleVars, std::map<std::string, float>& jetVars) {
        // make inputs for ParT with scaled features

        for (auto &v: input_feats_) {
            v.second.clear();
        }

        // fill input_feats_
        // existing features
        input_feats_["part_deta"].assign(particleVars["part_deta"].begin(), particleVars["part_deta"].end());
        input_feats_["part_dphi"].assign(particleVars["part_dphi"].begin(), particleVars["part_dphi"].end());
        input_feats_["part_charge"].assign(particleVars["part_charge"].begin(), particleVars["part_charge"].end());
        input_feats_["part_d0err"].assign(particleVars["part_d0err"].begin(), particleVars["part_d0err"].end());
        input_feats_["part_dzerr"].assign(particleVars["part_dzerr"].begin(), particleVars["part_dzerr"].end());

        for (size_t i = 0; i < particleVars["part_px"].size(); i++) {
            // calculating new features
            input_feats_["part_mask"].push_back(1);
            input_feats_["part_px_scale"].push_back(particleVars["part_px"][i] / jetVars["jet_pt"] * 500);
            input_feats_["part_py_scale"].push_back(particleVars["part_py"][i] / jetVars["jet_pt"] * 500);
            input_feats_["part_pz_scale"].push_back(particleVars["part_pz"][i] / jetVars["jet_pt"] * 500);
            input_feats_["part_energy_scale"].push_back(particleVars["part_energy"][i] / jetVars["jet_pt"] * 500);

            input_feats_["part_pt"].push_back(std::hypot(particleVars["part_px"][i], particleVars["part_py"][i]));
            input_feats_["part_pt_scale"].push_back(std::hypot(input_feats_["part_px_scale"][i], input_feats_["part_py_scale"][i]));
            input_feats_["part_pt_scale_log"].push_back(std::log(input_feats_["part_pt_scale"][i]));
            input_feats_["part_e_scale_log"].push_back(std::log(input_feats_["part_energy_scale"][i]));
            input_feats_["part_logptrel"].push_back(std::log(input_feats_["part_pt"][i] / jetVars["jet_pt"]));
            input_feats_["part_logerel"].push_back(std::log(particleVars["part_energy"][i] / jetVars["jet_energy"]));
            input_feats_["part_deltaR"].push_back(std::hypot(particleVars["part_deta"][i], particleVars["part_dphi"][i]));
            input_feats_["part_d0"].push_back(std::tanh(particleVars["part_d0val"][i]));
            input_feats_["part_dz"].push_back(std::tanh(particleVars["part_dzval"][i]));
            input_feats_["part_isElectron"].push_back(particleVars["part_pid"][i] == 11 || particleVars["part_pid"][i] == -11);
            input_feats_["part_isMuon"].push_back(particleVars["part_pid"][i] == 13 || particleVars["part_pid"][i] == -13);
            input_feats_["part_isPhoton"].push_back(particleVars["part_pid"][i] == 22);
            input_feats_["part_isChargedHadron"].push_back(particleVars["part_charge"][i] != 0 && !input_feats_["part_isElectron"][i] && !input_feats_["part_isMuon"][i]);
            input_feats_["part_isNeutralHadron"].push_back(particleVars["part_charge"][i] == 0 && !input_feats_["part_isPhoton"][i]);
        }

        // reset data_ to all zeros
        for (size_t i = 0; i < input_names_.size(); i++) {
            data_[i].assign(data_[i].size(), 0);
        }
        // construct the input data_
        for (size_t i = 0; i < input_names_.size(); i++) { // loop over input names
            for (int j = 0; j < input_shapes_[i][1]; j++) { // loop over channels

                auto name = std::get<0>(input_var_info_[i][j]);
                auto subtract_val = std::get<1>(input_var_info_[i][j]);
                auto multiply_val = std::get<2>(input_var_info_[i][j]);
                auto clip_min = std::get<3>(input_var_info_[i][j]);
                auto clip_max = std::get<4>(input_var_info_[i][j]);
                float val;

                int len = std::min((int)input_shapes_[i][2], (int)input_feats_[name].size());
                for (auto l = 0; l < len; l++) { // loop over particle length
                    data_[i][j * input_shapes_[i][2] + l] = std::clamp((input_feats_[name][l] - subtract_val) * multiply_val, clip_min, clip_max);
                }
            }
            if (debug_) {
                std::cout << "input: " << input_names_[i] << ":\n";
                for (int j = 0; j < input_shapes_[i][1]; j++) {
                    std::cout << "> var: " << std::get<0>(input_var_info_[i][j]) << ":\n";
                    for (int k = 0; k < input_shapes_[i][2]; k++) {
                        std::cout << data_[i][j * input_shapes_[i][2] + k] << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }

    }
};

#endif
