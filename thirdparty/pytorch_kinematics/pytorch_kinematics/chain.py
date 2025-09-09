import torch
from . import jacobian
import pytorch_kinematics.transforms as tf
from torch.func import vmap, jacrev


def skew_symmetric_matrix(vec: torch.Tensor) -> torch.Tensor:
    """Computes the skew-symmetric matrix of a vector.

    Args:
        vec: The input vector. Shape is (3,) or (N, 3).

    Returns:
        The skew-symmetric matrix. Shape is (1, 3, 3) or (N, 3, 3).

    Raises:
        ValueError: If input tensor is not of shape (..., 3).
    """
    # check input is correct
    if vec.shape[-1] != 3:
        raise ValueError(f"Expected input vector shape mismatch: {vec.shape} != (..., 3).")
    # unsqueeze the last dimension
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
    shape = vec.shape[:-1]
    length = torch.prod(torch.tensor(shape)).item()
    # create a skew-symmetric matrix
    skew_sym_mat = torch.zeros(length, 3, 3, device=vec.device, dtype=vec.dtype)
    skew_sym_mat[..., 0, 1] = -vec[..., 2]
    skew_sym_mat[..., 0, 2] = vec[..., 1]
    skew_sym_mat[..., 1, 2] = -vec[..., 0]
    skew_sym_mat[..., 1, 0] = vec[..., 2]
    skew_sym_mat[..., 2, 0] = -vec[..., 1]
    skew_sym_mat[..., 2, 1] = vec[..., 0]
    skew_sym_mat = skew_sym_mat.view(*shape, 3, 3)

    return skew_sym_mat


def ensure_2d_tensor(th, dtype, device):
    if not torch.is_tensor(th):
        th = torch.tensor(th, dtype=dtype, device=device)
    if len(th.shape) == 0:
        N = 1
        th = th.view(1, 1)
    elif len(th.shape) == 1:
        N = len(th)
        th = th.view(-1, 1)
    else:
        N = th.shape[0]
    return th, N


class Chain(object):
    def __init__(self, root_frame, dtype=torch.float32, device="cpu"):
        self._root = root_frame
        self.dtype = dtype
        self.device = device
        self._frame_names = []

        def _load_frames_recursive(frame):
            self._frame_names.append(frame.link.name)
            for child in frame.children:
                _load_frames_recursive(child)

        _load_frames_recursive(self._root)
        # print("Built chain with %d frames." % len(self._frame_names))
        # print("Root frame: %s" % self._root.name)
        # print("Chain:")
        # for f in self._frame_names:
        #     print("-", f)

    def to(self, dtype=None, device=None):
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        self._root = self._root.to(dtype=self.dtype, device=self.device)
        return self

    @property
    def tails(self):
        root = self._root
        tails = []
        children = [*root.children]
        while len(children) > 0:
            child = children.pop()
            if child.children is None or len(child.children) == 0:
                tails.append(child)
            else:
                children.extend(child.children)
        return tails

    def __str__(self):
        return str(self._root)

    @staticmethod
    def _find_frame_recursive(name, frame):
        for child in frame.children:
            if child.name == name:
                return child
            ret = Chain._find_frame_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_frame(self, name):
        if self._root.name == name:
            return self._root
        return self._find_frame_recursive(name, self._root)

    @staticmethod
    def _find_link_recursive(name, frame):
        for child in frame.children:
            if child.link.name == name:
                return child.link
            ret = Chain._find_link_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_link(self, name):
        if self._root.link.name == name:
            return self._root.link
        return self._find_link_recursive(name, self._root)

    @staticmethod
    def _get_joint_parameter_names(frame, exclude_fixed=True):
        joint_names = []
        if not (exclude_fixed and frame.joint.joint_type == "fixed"):
            joint_names.append(frame.joint.name)
        for child in frame.children:
            joint_names.extend(Chain._get_joint_parameter_names(child, exclude_fixed))
        return joint_names

    def get_joint_parameter_names(self, exclude_fixed=True):
        names = self._get_joint_parameter_names(self._root, exclude_fixed)
        return sorted(set(names), key=names.index)

    def add_frame(self, frame, parent_name):
        frame = self.find_frame(parent_name)
        if not frame is None:
            frame.add_child(frame)

    @staticmethod
    def _forward_kinematics(root, th_dict, world=tf.Transform3d(), parent=""):
        link_transforms = {}

        th, N = ensure_2d_tensor(th_dict.get(root.joint.name, 0.0), world.dtype, world.device)
        trans = world.compose(root.get_transform(th.view(N, 1)))
        link_transforms[root.link.name] = trans.compose(root.link.offset)

        for child in root.children:
            link_transforms.update(Chain._forward_kinematics(child, th_dict, trans, root.name))
        return link_transforms

    def forward_kinematics(self, th, world=tf.Transform3d()):
        if not isinstance(th, dict):
            jn = self.get_joint_parameter_names()
            if len(jn) != th.shape[1]:
                raise ValueError("Invalid number of joint parameters.", "Expected %d, got %d." % (len(jn), th.shape[1]))
            assert len(jn) == th.shape[1]
            th_dict = dict((j, th[:, i]) for i, j in enumerate(jn))
        else:
            th_dict = th
        if world.dtype != self.dtype or world.device != self.device:
            world = world.to(dtype=self.dtype, device=self.device, copy=True)
        return self._forward_kinematics(self._root, th_dict, world, "WORLD")

    @staticmethod
    def _fk_vectorized(root, theta: torch.Tensor, theta_idx: int, world: tf.Transform3d):
        batch_size = theta.shape[0]

        if root.joint.joint_type != "fixed":
            link_theta = theta[..., theta_idx].view(-1, 1)
            theta_idx += 1
        else:
            link_theta = torch.zeros(1, dtype=theta.dtype, device=theta.device).view(-1, 1)

        trans = world.compose(root.get_transform(link_theta))

        pose = trans.compose(root.link.offset).get_matrix()
        q_wxyz = tf.matrix_to_quaternion(pose[:, :3, :3])
        pose = torch.cat([pose[:, :3, -1], q_wxyz], dim=1)
        pose = pose.expand(batch_size, -1)

        frame_poses = [pose]

        if root.children is not None:
            for child in root.children:
                poses, theta_idx = Chain._fk_vectorized(child, theta, theta_idx, trans)
                frame_poses += poses

        return frame_poses, theta_idx

    @staticmethod
    def _fk_vectorized_jac(root, theta: torch.Tensor, theta_idx: int, world: tf.Transform3d, parent_jacobian: torch.Tensor):
        """

        Args:
            parent_jacobian: (B, 6, n_joints)
        """
        child_jacobian = parent_jacobian.clone()
        if root.joint.joint_type != "fixed":
            link_theta = theta[..., theta_idx].view(-1, 1)
            theta_idx += 1
            # axis = (roma.unitquat_to_rotmat(world.get_matrix()[:, :3, :3]) @ axis.unsqueeze(0).unsqueeze(-1)).squeeze(-1)s
        else:
            link_theta = torch.zeros(1, dtype=theta.dtype, device=theta.device).view(-1, 1)

        current_link_tf_local = root.get_transform(link_theta)
        current_link_tf_world = world.compose(current_link_tf_local)
        offset_world = current_link_tf_local.transform_normals(root.link.offset.get_matrix()[..., :3, -1]).squeeze(1)

        current_link_offset = world.transform_normals(current_link_tf_local.get_matrix()[..., :3, -1].unsqueeze(1)).squeeze(1)
        if current_link_offset.ndim == 2:
            current_link_offset = current_link_offset.unsqueeze(-1)

        child_jacobian[..., :3, :] = parent_jacobian[..., :3, :] + parent_jacobian[..., 3:, :].cross(current_link_offset, dim=-2)

        if root.joint.joint_type != "fixed":
            # convert into current link frame
            axis = current_link_tf_world.transform_normals(root.joint.axis.unsqueeze(0)).squeeze(1)
            if root.joint.joint_type == "revolute":
                child_jacobian[..., :3, theta_idx - 1] += axis.cross(offset_world, dim=-1)
                child_jacobian[..., 3:, theta_idx - 1] += axis
            elif root.joint.joint_type == "prismatic":
                child_jacobian[..., :3, theta_idx - 1] += axis

        jacobians = [child_jacobian]
        if root.children is not None:
            for child in root.children:
                child_jacs, theta_idx = Chain._fk_vectorized_jac(child, theta, theta_idx, current_link_tf_world, parent_jacobian=child_jacobian)
                jacobians += child_jacs

        return jacobians, theta_idx

    def jac_vectorized(self, th):
        """Returns the Jacobian of the chain at the given joint angles.

        Shape: (B, n_bodies, )
        """
        squeeze = False
        if th.ndim == 1:
            th = th.unsqueeze(0)
            squeeze = True
        jacobian = torch.zeros(th.shape[0], 6, th.shape[-1], dtype=th.dtype, device=th.device)
        data = self._fk_vectorized_jac(self._root, th, theta_idx=0, world=tf.Transform3d(device=self.device), parent_jacobian=jacobian)[0]
        data = torch.stack(data, dim=1)
        if squeeze:
            data = data.squeeze(0)
        return data

    def fk_vectorized(self, th):
        squeeze = False
        if th.ndim == 1:
            th = th.unsqueeze(0)
            squeeze = True
        data = self._fk_vectorized(self._root, th, theta_idx=0, world=tf.Transform3d(device=self.device))[0]
        data = torch.stack(data, dim=1)
        if squeeze:
            data = data.squeeze(0)
        return data

    def jacobian(self, th):
        jac = self.jac_vectorized(th)
        return jac

    def analytical_jacobian(self, th, analytical=False):
        squeeze = False
        if th.ndim == 1:
            th = th.unsqueeze(0)
            squeeze = True

        J = vmap(jacrev(self.fk_vectorized))(th + 1e-4)
        if analytical:
            return J if not squeeze else J.squeeze(0)

        fk = self.fk_vectorized(th)
        quat_wxyz = fk[..., 3:]
        # convert angular quaternion to angular velocity at the end effector
        quat_wxyz_dot = J[..., 3:, :]
        q0, q1, q2, q3 = quat_wxyz[..., 0], quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3]
        H_X_EE = torch.stack(
            [
                torch.stack([-q1, q0, -q3, q2], dim=-1),
                torch.stack([-q2, q3, q0, -q1], dim=-1),
                torch.stack([-q3, -q2, q1, q0], dim=-1),
            ],
            dim=-2,
        )

        ang_vel = 2 * (H_X_EE @ quat_wxyz_dot)

        J_geom = torch.cat([J[..., :3, :], ang_vel], dim=-2)
        # Replace NaN with zeros
        J_geom[torch.isnan(J_geom)] = 0.0
        J = J_geom
        if squeeze:
            J = J.squeeze(0)
        return J


class SerialChain(Chain):
    def __init__(self, chain, end_frame_name, root_frame_name="", **kwargs):
        if root_frame_name == "":
            super(SerialChain, self).__init__(chain._root, **kwargs)
        else:
            super(SerialChain, self).__init__(chain.find_frame(root_frame_name), **kwargs)
            if self._root is None:
                raise ValueError("Invalid root frame name %s." % root_frame_name)
        self._serial_frames = self._generate_serial_chain_recurse(self._root, end_frame_name)
        if self._serial_frames is None:
            raise ValueError("Invalid end frame name %s." % end_frame_name)

    @staticmethod
    def _generate_serial_chain_recurse(root_frame, end_frame_name):
        for child in root_frame.children:
            if child.name == end_frame_name:
                return [child]
            else:
                frames = SerialChain._generate_serial_chain_recurse(child, end_frame_name)
                if not frames is None:
                    return [child] + frames
        return None

    def get_joint_parameter_names(self, exclude_fixed=True):
        names = []
        for f in self._serial_frames:
            if exclude_fixed and f.joint.joint_type == "fixed":
                continue
            names.append(f.joint.name)
        return names

    def forward_kinematics(self, th, world=tf.Transform3d(), end_only=True):
        if world.dtype != self.dtype or world.device != self.device:
            world = world.to(dtype=self.dtype, device=self.device, copy=True)
        th, N = ensure_2d_tensor(th, self.dtype, self.device)

        cnt = 0
        link_transforms = {}
        trans = tf.Transform3d(matrix=world.get_matrix().repeat(N, 1, 1))
        for f in self._serial_frames:
            trans = trans.compose(f.get_transform(th[:, cnt].view(N, 1)))
            link_transforms[f.link.name] = trans.compose(f.link.offset)
            if f.joint.joint_type != "fixed":
                cnt += 1
        return link_transforms[self._serial_frames[-1].link.name] if end_only else link_transforms

    def jacobian(self, th, locations=None):
        if locations is not None:
            locations = tf.Transform3d(pos=locations)
        return jacobian.calc_jacobian(self, th, tool=locations)
