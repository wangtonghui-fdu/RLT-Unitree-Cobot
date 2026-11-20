"""mk_layout_v6.py OUT: layout for gen_dual_v6.py from the probe_dual6 scans (explicit choice, 11-20):
handover = ho6_sel.json[14]: charger flat at (0.14, -0.37, table + 0.22), tail 10 deg off straight-right, left finger sign -1 (the
sign of the ready posture -> Cartesian approach), right fingers fz +1; exactly 0 deg exists only 6-8 cm above the table.
charger (0.30, -0.345); strip centre (-0.135, -0.345) crosswise, plug turned the 'twin 1' way (Cartesian carry), hover 5.5 cm."""
import json, sys
HO = json.load(open("ho6_sel.json")); H = HO[14]; B1 = json.load(open("b1_6.json")); B2 = json.load(open("b2_6m1.json")); HM = json.load(open("ho6_m1.json"))
assert H["item"] == [0.14, -0.37, 0.22, -10, 1, -1], H["item"]
hm = next(i for i, h in enumerate(HM) if h["item"] == H["item"])
r1 = next(r for r in B1 if r["h"] == 14 and r["charger"] == [0.30, -0.33, 0] and r["ok"])
r2 = next(r for r in B2 if r["h"] == hm and r["strip"] == [-0.135, -0.345, 0] and r["twin"] == 1 and r["ok"])
L = dict(version="v6", table_z=0.75, charger=[0.30, -0.345, 0], strip=[-0.135, -0.345, 0], twin=1, h_hi=0.055,
         q_home=[0.968, 1.041, 1.738, 1.709, -0.506, -0.051, -0.824], q_preL=r1["q_preL"], q_handL=H["qL"], q_handR=H["qR_grasp"], q_preR=H["qR_pre"],
         q_insR=r2["q_insR"], left_up=H["left_up"], T_ch_H=H["T_ch"], fz_right=H["item"][4], fs_left=H["item"][5], y_lgrip=0.014, z_lgrip=-0.026, y_rgrip=0.036,
         hold=r2["hold"], probe=dict(handover=H, b1=r1, b2=r2))
json.dump(L, open(sys.argv[1], "w"), indent=1)
print("layout:", L["charger"], L["strip"], "handover", H["item"], "hold", L["hold"]["par"], "margins", r1["margin"], r2["margin"], r2["hold"]["margin"], "cart carry", not r2["joint_carry"])
