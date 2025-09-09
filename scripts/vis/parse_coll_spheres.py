import json
data = json.load(open('/home/zrene/git/DexGraspNet/graspqp/assets/ability_hand/penetration_points.json'))
link_mapping = {
    # "index_link_1": "index_link_1/contact",
    # "index_link_2": "index_link_2/contact",
    # "index_link_3": "index_link_3/contact",
    # "middle_link_1": "middle_link_1/contact",
    # "middle_link_2": "middle_link_2/contact",
    # "middle_link_3": "middle_link_3/contact",
    # "ring_link_1": "ring_link_1/contact",
    # "ring_link_2": "ring_link_2/contact",
    # "ring_link_3": "ring_link_3/contact",
    # "thumb_link_2": "thumb_link_2/contact",
    # "thumb_link_3": "thumb_link_3/contact",
    # "palm_link": "palm_link/contact",
}

configs = []
for entry in data:
    link_name = link_mapping.get(entry, entry)
    value = data[entry]
    
    spheres_cfg = []
    
    for sphere in value:
        if len(sphere) == 4:
            x,y,z,r = sphere
        else:
            x,y,z = sphere
            r = 0.01 # default radius
        config = f"""MeshTrackerCfg.MeshTargetCfg.CollSphereCfg(radius={r:.3f}, pos=[{x:.3f}, {y:.3f}, {z:.3f}])"""
        spheres_cfg.append(config)
        
    config_entry = f"""
    MeshTrackerCfg.MeshTargetCfg(
        target_prim_expr="/World/envs/env_.*/Robot/{link_name}/contact",
        is_robot_link=True,
        contact_link=True,
        n_pts=n_finger_pts_collision,
        spheres=[
            {','.join(spheres_cfg)}
        ],
    )"""
    configs.append(config_entry)
    
print(",".join(configs))