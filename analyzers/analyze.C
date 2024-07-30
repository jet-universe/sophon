#include <iostream>
#include <unordered_set>
#include <utility>
#include "TClonesArray.h"
#include "classes/DelphesClasses.h"
#include "ExRootAnalysis/ExRootTreeReader.h"
#include "OrtHelper.h"
#include "ParticleInfo.h"


void analyze(TString inputFile, TString outputFile, TString modelPath, TString jetBranch = "JetPUPPIAK8", bool debug = false) {

    TFile *fout = new TFile(outputFile, "RECREATE");
    TTree *tree = new TTree("tree", "tree");

    // Define output branches
    std::map<std::string, std::vector<float>* > outputVars;
    outputVars["jet_pt"] = new std::vector<float>;
    for (int i = 0; i < 188; i++) {
        outputVars["jet_probs_" + std::to_string(i)] = new std::vector<float>;
    }
    for (int i = 0; i < 128; i++) {
        outputVars["jet_hidneurons_" + std::to_string(i)] = new std::vector<float>;
    }

    // Set output branches
    for (auto &var : outputVars) {
        tree->Branch(var.first.c_str(), &var.second, /*bufsize=*/102400);
    }

    // Read input
    TChain *chain = new TChain("Delphes");
    chain->Add(inputFile);
    ExRootTreeReader *treeReader = new ExRootTreeReader(chain);
    Long64_t allEntries = treeReader->GetEntries();

    std::cerr << "** Input file: " << inputFile << std::endl;
    std::cerr << "** Jet branch: " << jetBranch << std::endl;
    std::cerr << "** Total events: " << allEntries << std::endl;

    // Analyze
    TClonesArray *branchVertex = treeReader->UseBranch("Vertex"); // used for pileup
    TClonesArray *branchParticle = treeReader->UseBranch("Particle");
    TClonesArray *branchPFCand = treeReader->UseBranch("ParticleFlowCandidate");
    TClonesArray *branchJet = treeReader->UseBranch(jetBranch);

    double jetR = jetBranch.Contains("AK15") ? 1.5 : 0.8;
    std::cerr << "jetR = " << jetR << std::endl;

    // Initialize onnx helper
    OrtHelper orthelper = OrtHelper(modelPath.Data(), debug);

    // Loop over all events
    allEntries = 10;
    for (Long64_t entry = 0; entry < allEntries; ++entry) {
        if (entry % 100 == 0) {
            std::cerr << "processing " << entry << " of " << allEntries << " events." << std::endl;
        }
        // Clear variables
        for (auto &var : outputVars) {
            var.second->clear();
        }

        // Load selected branches with data from specified event
        treeReader->ReadEntry(entry);

        // Loop over all jets in event
        for (Int_t i = 0; i < branchJet->GetEntriesFast(); ++i) {
            const Jet *jet = (Jet *)branchJet->At(i);

            // Initialize the input variables to infer the model
            std::map<std::string, std::vector<float>> particleVars;
            std::map<std::string, float> jetVars;

            jetVars["jet_pt"] = jet->PT;
            jetVars["jet_eta"] = jet->Eta;
            jetVars["jet_phi"] = jet->Phi;
            jetVars["jet_energy"] = jet->P4().Energy();

            jetVars["jet_sdmass"] = jet->SoftDroppedP4[0].M();
            jetVars["jet_trmass"] = jet->TrimmedP4[0].M();
            jetVars["jet_tau1"] = jet->Tau[0];
            jetVars["jet_tau2"] = jet->Tau[1];
            jetVars["jet_tau3"] = jet->Tau[2];
            jetVars["jet_tau4"] = jet->Tau[3];

            for (const auto &name : {"part_px", "part_py", "part_pz", "part_energy", "part_pt", "part_deta", "part_dphi", "part_charge", "part_pid", "part_d0val", "part_d0err", "part_dzval", "part_dzerr"}) {
                particleVars[name] = std::vector<float>();
            }

            // Loop over all jet's constituents
            std::vector<ParticleInfo> particles;
            for (Int_t j = 0; j < jet->Constituents.GetEntriesFast(); ++j) {
                const TObject *object = jet->Constituents.At(j);

                // Check if the constituent is accessible
                if (!object)
                continue;

                if (object->IsA() == GenParticle::Class()) {
                    particles.emplace_back((GenParticle *)object);
                } else if (object->IsA() == ParticleFlowCandidate::Class()) {
                    particles.emplace_back((ParticleFlowCandidate *)object);
                }
                const auto &p = particles.back();
                if (std::abs(p.pz) > 10000 || std::abs(p.eta) > 5 || p.pt <= 0) {
                    particles.pop_back();
                }
            }

            // Sort particles by pt
            std::sort(particles.begin(), particles.end(), [](const auto &a, const auto &b) { return a.pt > b.pt; });

            // Load the primary vertex
            const Vertex *pv = (branchVertex != nullptr) ? ((Vertex *)branchVertex->At(0)) : nullptr;

            for (const auto &p : particles) {
                particleVars["part_px"].push_back(p.px);
                particleVars["part_py"].push_back(p.py);
                particleVars["part_pz"].push_back(p.pz);
                particleVars["part_energy"].push_back(p.energy);
                particleVars["part_pt"].push_back(p.pt);
                particleVars["part_deta"].push_back((jet->Eta > 0 ? 1 : -1) * (p.eta - jet->Eta));
                particleVars["part_dphi"].push_back(deltaPhi(p.phi, jet->Phi));
                particleVars["part_charge"].push_back(p.charge);
                particleVars["part_pid"].push_back(p.pid);
                particleVars["part_d0val"].push_back(p.d0);
                particleVars["part_d0err"].push_back(p.d0err);
                particleVars["part_dzval"].push_back((pv && p.dz != 0) ? (p.dz - pv->Z) : p.dz);
                particleVars["part_dzerr"].push_back(p.dzerr);
            }

            // Infer the Sophon model
            orthelper.infer_model(particleVars, jetVars);
            const auto &output = orthelper.get_output();

            // Get inference output
            for (size_t i = 0; i < 188; i++) {
                outputVars["jet_probs_" + std::to_string(i)]->push_back(output[i]);
            }
            for (size_t i = 0; i < 128; i++) {
                outputVars["jet_hidneurons_" + std::to_string(i)]->push_back(output[i + 188]);
            }
            outputVars["jet_pt"]->push_back(jet->PT);
        } // end loop of jets

        tree->Fill();

    } // end loop of events

    tree->Write();
    std::cerr << TString::Format("** Processed %d events **", int(allEntries)) << std::endl;

    delete treeReader;
    delete chain;
    delete fout;
}
