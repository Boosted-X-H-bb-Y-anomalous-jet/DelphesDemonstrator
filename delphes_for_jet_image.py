import uproot
import numpy as np
import h5py
import sys

nMaxPreselected = 3000

def deltaR(eta1, phi1, eta2, phi2):
    """Calculate delta R between two particles."""
    dphi = np.abs(phi1 - phi2)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return np.sqrt((eta1 - eta2) ** 2 + dphi ** 2)

def select_and_convert(file_name, output_file):
    """
    Select and convert jet data from a ROOT file using uproot.
    
    Parameters:
    -----------
    file_name : str
        Path to the input ROOT file
    output_file : str
        Path to the output HDF5 file
    
    Returns:
    --------
    int
        Total number of processed events
    """
    with uproot.open(file_name) as root_file:
        tree = root_file["Delphes"]
        
        jet_kinematics_data = []
        jet1_pfcands_data = []
        jet2_pfcands_data = []
        h_jet_idx_data = []
        
        nTotal = 0
        nPreselected = 0
        
        for batch in tree.iterate(
            ['FatJet.PT', 'FatJet.Eta', 'FatJet.Phi', 'FatJet.Mass', 
             'ParticleFlowCandidate.PT', 'ParticleFlowCandidate.Eta', 
             'ParticleFlowCandidate.Phi', 'ParticleFlowCandidate.Mass',
             'GenParticle.Rapidity', 'GenParticle.Phi', 'GenParticle.PID'],
            library='np'
        ):
            fat_jet_pt = batch['FatJet.PT']
            fat_jet_eta = batch['FatJet.Eta']
            fat_jet_phi = batch['FatJet.Phi']
            fat_jet_mass = batch['FatJet.Mass']
            
            pf_cand_pt = batch['ParticleFlowCandidate.PT']
            pf_cand_eta = batch['ParticleFlowCandidate.Eta']
            pf_cand_phi = batch['ParticleFlowCandidate.Phi']
            pf_cand_mass = batch['ParticleFlowCandidate.Mass']
            
            gen_particle_eta = batch['GenParticle.Rapidity']
            gen_particle_phi = batch['GenParticle.Phi']
            gen_particle_pid = batch['GenParticle.PID']
            
            for event_idx in range(len(fat_jet_pt)):
                if nPreselected >= nMaxPreselected:
                    break
                
                nTotal += 1
                
                if len(fat_jet_pt[event_idx]) < 2:
                    continue
                
                higgs_indices = np.where(gen_particle_pid[event_idx] == 25)[0]
                if len(higgs_indices) == 0:
                    continue
                
                # Take the first Higgs boson if multiple exist
                higgs_idx = higgs_indices[0]
                higgs_eta = gen_particle_eta[event_idx][higgs_idx]
                higgs_phi = gen_particle_phi[event_idx][higgs_idx]
                
                pt1, pt2 = fat_jet_pt[event_idx][:2]
                eta1, eta2 = fat_jet_eta[event_idx][:2]
                phi1, phi2 = fat_jet_phi[event_idx][:2]
                mass1, mass2 = fat_jet_mass[event_idx][:2]
                
                dr1 = deltaR(higgs_eta, higgs_phi, eta1, phi1)
                dr2 = deltaR(higgs_eta, higgs_phi, eta2, phi2)
                
                deta = abs(eta1 - eta2)
                
                px1 = pt1 * np.cos(phi1)
                py1 = pt1 * np.sin(phi1)
                pz1 = pt1 * np.sinh(eta1)
                e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + mass1**2)
                
                px2 = pt2 * np.cos(phi2)
                py2 = pt2 * np.sin(phi2)
                pz2 = pt2 * np.sinh(eta2)
                e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + mass2**2)
                
                mjj = np.sqrt((e1 + e2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2)
                
                if not (pt1 > 300 and pt2 > 300 and deta < 1.3 and mjj > 1300 and abs(eta1)<2.5 and abs(eta2)<2.5):
                    continue
                
                # Determine which jet is closer to Higgs and apply mass cut
                if dr1 < dr2:
                    if not (mass1 > 100 and mass1 < 150):
                        continue
                    h_jet_idx = 0
                else:
                    if not (mass2 > 100 and mass2 < 150):
                        continue
                    h_jet_idx = 1
                
                nPreselected += 1
                
                # Placeholder for third jet (to satisfy data format)
                pt3, eta3, phi3, msoftdrop3 = 0, 0, 0, 0
                jet_kinematics = [mjj, deta, pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2, pt3, eta3, phi3, msoftdrop3]
                
                jet1_pfcands = []
                jet2_pfcands = []
                
                for j in range(len(pf_cand_pt[event_idx])):
                    pt = pf_cand_pt[event_idx][j]
                    eta = pf_cand_eta[event_idx][j]
                    phi = pf_cand_phi[event_idx][j]
                    mass = pf_cand_mass[event_idx][j]

                    if abs(eta)>4.:
                        continue
                    
                    dr1 = deltaR(eta, phi, eta1, phi1)
                    dr2 = deltaR(eta, phi, eta2, phi2)
                    
                    px = pt * np.cos(phi)
                    py = pt * np.sin(phi)
                    pz = pt * np.sinh(eta)

                    e = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
                    
                    if dr1 < 0.8 and len(jet1_pfcands) < 100:
                        jet1_pfcands.append([px, py, pz, e])
                    elif dr2 < 0.8 and len(jet2_pfcands) < 100:
                        jet2_pfcands.append([px, py, pz, e])
                
                # Pad candidates to ensure consistent length
                while len(jet1_pfcands) < 100:
                    jet1_pfcands.append([0.0, 0.0, 0.0, 0.0])
                while len(jet2_pfcands) < 100:
                    jet2_pfcands.append([0.0, 0.0, 0.0, 0.0])
                
                jet_kinematics_data.append(jet_kinematics)
                jet1_pfcands_data.append(jet1_pfcands)
                jet2_pfcands_data.append(jet2_pfcands)
                h_jet_idx_data.append(h_jet_idx)
        
        jet_kinematics_data = np.array(jet_kinematics_data, dtype=np.float32)
        jet1_pfcands_data = np.array(jet1_pfcands_data, dtype=np.float32)
        jet2_pfcands_data = np.array(jet2_pfcands_data, dtype=np.float32)
        h_jet_idx_data = np.array(h_jet_idx_data, dtype=np.int8)
        
        with h5py.File(output_file, "w") as h5f:
            h5f.create_dataset("nProcessed", data=nTotal)
            h5f.create_dataset("jet_kinematics", data=jet_kinematics_data, chunks=True, compression='gzip')
            h5f.create_dataset("jet1_PFCands", data=jet1_pfcands_data, chunks=True, maxshape=(None, 100, 4), compression='gzip')
            h5f.create_dataset("jet2_PFCands", data=jet2_pfcands_data, chunks=True, maxshape=(None, 100, 4), compression='gzip')
            h5f.create_dataset("H_jet_idx", data=h_jet_idx_data, chunks=True, compression='gzip')
    
    return nTotal

if __name__ == "__main__":
    file_name = sys.argv[1]
    output_file = file_name.replace(".root",".h5")
    select_and_convert(file_name, output_file)