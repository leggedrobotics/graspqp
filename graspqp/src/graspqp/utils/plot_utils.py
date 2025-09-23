import plotly.graph_objs as go


def show_initialization(object_model, hand_model, batch_size, n_objects):
    # create open3d viewer
    import open3d as o3d
    from open3d.visualization.rendering import MaterialRecord

    for object_idx in range(n_objects):
        object_mesh = object_model.get_open3d_data(object_idx * batch_size)
        geoms = []
        geoms.append(
            {
                "name": "object",
                "geometry": object_mesh,
            }
        )

        for i in range(batch_size):
            idx = object_idx * batch_size + i
            hand_mesh = hand_model.get_open3d_data(idx)
            hand_mesh.compute_vertex_normals()
            # add alpha channel to mesh
            # make it transparent

            mat_box = MaterialRecord()
            # mat_box.shader = 'defaultLitTransparency'
            mat_box.shader = "defaultLitSSR"
            mat_box.base_color = [1.0, 0.0, 0.0, 0.99]
            mat_box.base_roughness = 0.0
            mat_box.base_reflectance = 0.0
            mat_box.base_clearcoat = 1.0
            mat_box.thickness = 1.0
            mat_box.transmission = 0.05
            mat_box.absorption_distance = 10
            mat_box.absorption_color = [0.1, 0.8, 0.8]
            geoms.append(
                {
                    "name": f"hand_{i}",
                    "geometry": hand_mesh,
                    # "material": mat_box
                }
            )

        o3d.visualization.draw(geoms, eye=(0.75, 0.75, 0.75), lookat=(0, 0, 0), up=(0, 0, 1))


def get_plotly_fig(object_model, hand_model, env_idx):
    object_plotly = object_model.get_plotly_data(env_idx, opacity=1.0, simplify=False)
    hand_plotly = hand_model.get_plotly_data(
        env_idx, opacity=0.9, with_contact_points=True, with_surface_points=False, with_penetration_points=False, simplify=True
    )

    data = object_plotly + hand_plotly
    # get contact points and normals
    batch_size, n_contact, _ = hand_model.get_contact_points().shape
    contact_hand_normals = hand_model.contact_normals

    device = object_model.device
    distance, contact_normal, closest_pts = object_model.cal_distance(hand_model.get_contact_points(), with_closest_points=True)
    # show contacts nromals
    normal = contact_normal[env_idx].detach().cpu().numpy() * 0.05
    contact_points = closest_pts[env_idx].detach().cpu().numpy()
    data = object_plotly + hand_plotly
    # Surface points of object

    # E_pen
    # object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
    # object_surface_points = (
    #     object_model.surface_points_tensor * object_scale
    # )  # (n_objects * batch_size_each, num_samples, 3)

    # distances = hand_model.cal_distance(object_surface_points)
    # distance = distances[env_idx].detach().cpu().numpy()
    # object_surface_points = object_surface_points[env_idx].detach().cpu().numpy()
    # free_pts = object_surface_points[distance > 0]

    # data.append(
    #     go.Scatter3d(
    #        x=free_pts[:, 0],
    #           y=free_pts[:, 1],
    #             z=free_pts[:, 2],
    #             mode="markers",
    #             marker=dict(size=5, color="red"),
    #             name="free points",
    #             legendgroup="free points",
    #             showlegend=False,

    #     )
    # )
    # occupied_pts = object_surface_points[distance <= 0]
    # data.append(
    #     go.Scatter3d(
    #        x=occupied_pts[:, 0],
    #           y=occupied_pts[:, 1],
    #             z=occupied_pts[:, 2],
    #             mode="markers",
    #             marker=dict(size=5, color="blue"),
    #             name="occupied points",
    #             legendgroup="occupied points",
    #             showlegend=False,

    #     )
    # )

    # Plot all points
    # distances[distances <= 0] = 0

    contact_candidates = hand_model.get_contact_candidates()[env_idx].detach().cpu().numpy()
    data.append(
        go.Scatter3d(
            x=contact_candidates[:, 0],
            y=contact_candidates[:, 1],
            z=contact_candidates[:, 2],
            mode="markers",
            marker=dict(size=5, color="blue"),
        )
    )

    # scatter plots for contacts, same legend gorup
    data.append(
        go.Scatter3d(
            x=contact_points[:, 0],
            y=contact_points[:, 1],
            z=contact_points[:, 2],
            mode="markers",
            marker=dict(size=5, color="green"),
            name="contact points",
            legendgroup="contact points",
            showlegend=False,
        )
    )
    for i in range(n_contact):
        data.append(
            go.Scatter3d(
                x=[
                    contact_points[i, 0],
                    contact_points[i, 0] + normal[i, 0],
                ],
                y=[
                    contact_points[i, 1],
                    contact_points[i, 1] + normal[i, 1],
                ],
                z=[
                    contact_points[i, 2],
                    contact_points[i, 2] + normal[i, 2],
                ],
                mode="lines",
                line=dict(color="green", width=5),
                legendgroup="contact points",
                showlegend=False,
            )
        )

    return go.Figure(data)
