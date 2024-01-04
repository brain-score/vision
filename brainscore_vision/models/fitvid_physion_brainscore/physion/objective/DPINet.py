import os
import pickle
import numpy as np
import scipy
import logging
import mlflow
import cv2
import vispy.scene
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from physopt.objective import PretrainingObjectiveBase, ExtractionObjectiveBase, ReadoutObjectiveBase
from physopt.objective import utils
from physion.objective.objective import PytorchModel
from physion.models.particle import GNSRigidH
from physion.data.flexdata import PhysicsFleXDataset, collate_fn, load_data_dominoes, \
    correct_bad_chair, remove_large_obstacles, subsample_particles_on_large_objects, \
    recalculate_velocities, prepare_input

use_gpu = torch.cuda.is_available()
class DPINetModel(PytorchModel):
    def get_model(self):
        args = self.pretraining_cfg.MODEL.args
        model = GNSRigidH(args, residual=True, use_gpu=use_gpu)
        if use_gpu:
            model = model.cuda()
        return model

    def load_model(self, model_file):
        checkpoint = torch.load(model_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        for attr in ['optimizer', 'scheduler']:
            if hasattr(self, attr) and attr+'_state_dict' in checkpoint:
                getattr(self, attr).load_state_dict(checkpoint[attr+'_state_dict'])
            else:
                logging.info(f'Not loading {attr}')
        return self.model

    def save_model(self, model_file):
        logging.info(f'Saved model checkpoint to: {model_file}')
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()},
                    model_file)

class PretrainingObjective(DPINetModel, PretrainingObjectiveBase):
    def get_pretraining_dataloader(self, datapaths, train):
        args = self.pretraining_cfg.DATA.args
        shuffle = True if train else False
        phase = 'train' if train else 'valid'
        dataset = PhysicsFleXDataset(datapaths, args, phase, args.verbose_data)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.pretraining_cfg.BATCH_SIZE,
            shuffle=shuffle, collate_fn=collate_fn)
        logging.info(f'Pretraining {phase} dataloader len: {len(dataloader)}')
        return dataloader

    def setup(self):
        super().setup()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.pretraining_cfg.TRAIN.LR, betas=(self.pretraining_cfg.TRAIN.args.beta1, 0.999))
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.8, patience=3, verbose=True) # TODO: step scheduler at every epoch
        self.criterionMSE = nn.MSELoss()

    def train_step(self, data):
        args = self.pretraining_cfg.TRAIN.args
        self.model.train()
        self.optimizer.zero_grad()
        attr, state, rels, n_particles, n_shapes, instance_idx, label, phases_dict_current= data
        Ra, node_r_idx, node_s_idx, pstep, rels_types = rels[3], rels[4], rels[5], rels[6], rels[7]
        Rr, Rs, Rr_idxs = [], [], []
        for j in range(len(rels[0])):
            Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]

            Rr_idxs.append(Rr_idx)
            Rr.append(torch.sparse.FloatTensor(
                Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))

            Rs.append(torch.sparse.FloatTensor(
                Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))

        data = [attr, state, Rr, Rs, Ra, Rr_idxs, label]

        with torch.set_grad_enabled(True):
            if use_gpu:
                for d in range(len(data)):
                    if type(data[d]) == list:
                        for t in range(len(data[d])):
                            data[d][t] = Variable(data[d][t].cuda())
                    else:
                        data[d] = Variable(data[d].cuda())
            else:
                for d in range(len(data)):
                    if type(data[d]) == list:
                        for t in range(len(data[d])):
                            data[d][t] = Variable(data[d][t])
                    else:
                        data[d] = Variable(data[d])

            attr, state, Rr, Rs, Ra, Rr_idxs, label = data

            predicted = self.model(
                attr, state, Rr, Rs, Ra, Rr_idxs, n_particles,
                node_r_idx, node_s_idx, pstep, rels_types,
                instance_idx, phases_dict_current, args.verbose_model)

        loss = self.criterionMSE(predicted, label) / args.forward_times
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self, data):
        args = self.pretraining_cfg.TRAIN.args
        self.model.train(False)
        attr, state, rels, n_particles, n_shapes, instance_idx, label, phases_dict_current= data
        Ra, node_r_idx, node_s_idx, pstep, rels_types = rels[3], rels[4], rels[5], rels[6], rels[7]
        Rr, Rs, Rr_idxs = [], [], []
        for j in range(len(rels[0])):
            Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]

            Rr_idxs.append(Rr_idx)
            Rr.append(torch.sparse.FloatTensor(
                Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))

            Rs.append(torch.sparse.FloatTensor(
                Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))

        data = [attr, state, Rr, Rs, Ra, Rr_idxs, label]

        with torch.set_grad_enabled(False):
            if use_gpu:
                for d in range(len(data)):
                    if type(data[d]) == list:
                        for t in range(len(data[d])):
                            data[d][t] = Variable(data[d][t].cuda())
                    else:
                        data[d] = Variable(data[d].cuda())
            else:
                for d in range(len(data)):
                    if type(data[d]) == list:
                        for t in range(len(data[d])):
                            data[d][t] = Variable(data[d][t])
                    else:
                        data[d] = Variable(data[d])

            attr, state, Rr, Rs, Ra, Rr_idxs, label = data

            # st_time = time.time()
            predicted = self.model(
                attr, state, Rr, Rs, Ra, Rr_idxs, n_particles,
                node_r_idx, node_s_idx, pstep, rels_types,
                instance_idx, phases_dict_current, args.verbose_model)

        loss = self.criterionMSE(predicted, label) / args.forward_times
        return {'val_loss': loss.item()}

class ExtractionObjective(DPINetModel, ExtractionObjectiveBase):
    def get_readout_dataloader(self, datapaths): # gets list of trials from labels .txt file
        pass

    def extract_feat_step(self): # hack since overriding 'call'
        pass

    @staticmethod
    def get_max_timestep(scenario): # TODO: dependent on fpt I think
        if scenario == "Support":
            max_timestep = 205
        elif scenario == "Link":
            max_timestep = 140
        elif scenario == "Contain":
            max_timestep = 125
        elif scenario in ["Collide", "Drape"]:
            max_timestep = 55
        else:
            max_timestep = 105
        return max_timestep

    @staticmethod
    def get_red_yellow_id(phases_dict, scenario, trial_name):
        if scenario in ["Dominoes", "Collide", "Drop"]:
            red_id = 1
            yellow_id = 0
        elif scenario in ["Drape"]:
            instance_idx = phases_dict["instance_idx"]
            yellow_id = 0
            red_id = len(instance_idx) - 1 -1
        elif scenario in ["Roll"]:
            yellow_id = 0
            if "ramp" in trial_name:
                red_id = 2
            else:
                red_id = 1
        else:
            if "red_id" not in phases_dict:
                print(arg_name, trial_id_name)
            red_id = phases_dict["red_id"]
            yellow_id = phases_dict["yellow_id"]
        return red_id, yellow_id

    def call(self, args):
        feature_file = utils.get_feats_from_artifact_store('test', self.tracking_uri, self.run_id, self.output_dir)
        if feature_file is not None: # features already extracted
            return 

        self.model.eval()
        scenario = self.readout_name
        args = self.pretraining_cfg.DATA.args
        dt = args.training_fpt * args.dt

        label_file = os.path.join(args.dpi_data_dir, 'test/labels', scenario + '.txt') # TODO: not robust
        gt_labels = []
        with open(label_file, "r") as f:
            for line in f:
                trial_name, label = line.strip().split(",")
                gt_labels.append((trial_name[:-5], (label == "True")))

        labels = []
        stimulus_name = []
        input_states = []
        observed_states = []
        simulated_states = []
        for trial_id, trial_cxt in enumerate(gt_labels):
            print("Rollout %d / %d" % (trial_id, len(gt_labels)))

            trial_name, label_gt = trial_cxt
            stimulus_name.append(trial_name)
            labels.append(label_gt)
            print(f'Trial name: {trial_name} ({label_gt})')
            trial_dir = os.path.join(args.dpi_data_dir, 'test', scenario, trial_name)

            time_step = len([fn for fn in os.listdir(trial_dir) if re.match('\d+\.h5', fn) is not None])
            timesteps  = [t for t in range(0, time_step - int(args.training_fpt), int(args.training_fpt))]
            total_nframes = len(timesteps)
            max_timestep = self.get_max_timestep(scenario)

            pkl_path = os.path.join(trial_dir, 'phases_dict.pkl')
            with open(pkl_path, "rb") as f:
                phases_dict = pickle.load(f)
            phases_dict["trial_dir"] = trial_dir

            red_id, yellow_id = self.get_red_yellow_id(phases_dict, scenario, trial_name)

            is_bad_chair = correct_bad_chair(phases_dict)
            is_remove_obstacles = remove_large_obstacles(phases_dict) # remove obstacles that are too big
            is_subsample = subsample_particles_on_large_objects(phases_dict, limit=args.subsample) # downsample large object

            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            gt_vid_path = os.path.join(self.output_dir, f'gt_{trial_name}.avi')
            gt_out = cv2.VideoWriter(gt_vid_path, fourcc, 20, (800, 600))
            pred_vid_path = os.path.join(self.output_dir, f'pred_{trial_name}.avi')
            pred_out = cv2.VideoWriter(pred_vid_path, fourcc, 20, (800, 600))

            particle_size = 6.0
            n_instance = 5 #args.n_instance
            c = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
            view = c.central_widget.add_view()
            if "Collide" in trial_name:
                distance = 6.0
            elif "Support" in trial_name:
                distance = 6.0 #6.0
            elif "Link" in trial_name:
                distance = 10.0
            elif "Drop" in trial_name:
                distance = 5.0
            elif "Drape" in trial_name:
                distance = 5.0
            else:
                distance = 3.0
            view.camera = vispy.scene.cameras.TurntableCamera(fov=50, azimuth=80, elevation=30, distance=distance, up='+y')
            n_instance = len(phases_dict["instance"])
            instance_colors = create_instance_colors(n_instance)
            colors = convert_groups_to_colors(
                phases_dict["instance_idx"],
                instance_colors=instance_colors)
            n_particle = phases_dict["instance_idx"][-1]
            add_floor(view)
            p1 = vispy.scene.visuals.Markers()
            p1.antialias = 0  # remove white edge
            floor_pos = np.array([[0, -0.5, 0]])
            line = vispy.scene.visuals.Line() 
            view.add(p1)
            view.add(line)

            start_timestep = 45 # start_id * training_fpt
            start_id = 15 
            assert start_timestep == start_id * args.training_fpt
            input_state = []
            gt_node_rs_idxs = []
            for current_fid, step in enumerate(timesteps[:start_id]):
                data_path = os.path.join(trial_dir, str(step) + '.h5')
                data_nxt_path = os.path.join(trial_dir, str(step + int(args.training_fpt)) + '.h5')

                data_names = ['positions', 'velocities']
                data = load_data_dominoes(data_names, data_path, phases_dict)

                data_nxt = load_data_dominoes(data_names, data_nxt_path, phases_dict)

                data_prev_path = os.path.join(trial_dir, str(max(0, step - int(args.training_fpt))) + '.h5')
                data_prev = load_data_dominoes(data_names, data_prev_path, phases_dict)

                _, data, data_nxt = recalculate_velocities([data_prev, data, data_nxt], dt, data_names)

                attr, state, rels, n_particles, n_shapes, instance_idx = \
                        prepare_input(data, args, phases_dict, args.verbose_data)

                Ra, node_r_idx, node_s_idx, pstep, rels_types = rels[3], rels[4], rels[5], rels[6], rels[7]
                gt_node_rs_idxs.append(np.stack([rels[0][0][0], rels[1][0][0]], axis=1))

                velocities_nxt = data_nxt[1]

                if step == 0:
                    positions, velocities = data
                    n_particles = positions.shape[0]
                    print("n_particles", n_particles)
                    clusters = phases_dict["clusters"]

                    p_gt = np.zeros((total_nframes, n_particles, args.position_dim))
                    v_nxt_gt = np.zeros((total_nframes, n_particles, args.position_dim))
                    p_pred = np.zeros((max_timestep, n_particles, args.position_dim))

                p_gt[current_fid] = positions[:, -args.position_dim:]
                v_nxt_gt[current_fid] = velocities_nxt[:, -args.position_dim:]

                positions = data[0]

                st, ed = instance_idx[red_id], instance_idx[red_id + 1]
                red_pts = positions[st:ed]

                st2, ed2 = instance_idx[yellow_id], instance_idx[yellow_id + 1]
                yellow_pts = positions[st2:ed2]

                input_state.append([red_pts, yellow_pts])

                p1.set_data(p_gt[current_fid, :n_particle], size=particle_size, edge_color='black', face_color=colors)
                line.set_data(pos=np.concatenate([p_gt[current_fid, :], floor_pos], axis=0), connect=gt_node_rs_idxs[current_fid])
                img = c.render()
                gt_out.write(img[:,:,:3])
                pred_out.write(img[:,:,:3])

            # add some black frames 
            for _ in range(5):
                gt_out.write(np.zeros((600,800,3), dtype=np.uint8))
                pred_out.write(np.zeros((600,800,3), dtype=np.uint8))

            # gt rollout
            observed_state = []
            for current_fid, step in enumerate(timesteps[start_id:]):
                data_path = os.path.join(trial_dir, str(step) + '.h5')
                data_nxt_path = os.path.join(trial_dir, str(step + int(args.training_fpt)) + '.h5')

                data_names = ['positions', 'velocities']
                data = load_data_dominoes(data_names, data_path, phases_dict)

                data_nxt = load_data_dominoes(data_names, data_nxt_path, phases_dict)

                data_prev_path = os.path.join(trial_dir, str(max(0, step - int(args.training_fpt))) + '.h5')
                data_prev = load_data_dominoes(data_names, data_prev_path, phases_dict)

                _, data, data_nxt = recalculate_velocities([data_prev, data, data_nxt], dt, data_names)

                attr, state, rels, n_particles, n_shapes, instance_idx = \
                        prepare_input(data, args, phases_dict, args.verbose_data)

                Ra, node_r_idx, node_s_idx, pstep, rels_types = rels[3], rels[4], rels[5], rels[6], rels[7]
                gt_node_rs_idxs.append(np.stack([rels[0][0][0], rels[1][0][0]], axis=1))

                velocities_nxt = data_nxt[1]

                p_gt[current_fid] = positions[:, -args.position_dim:]
                v_nxt_gt[current_fid] = velocities_nxt[:, -args.position_dim:]

                positions = data[0]

                st, ed = instance_idx[red_id], instance_idx[red_id + 1]
                red_pts = positions[st:ed]

                st2, ed2 = instance_idx[yellow_id], instance_idx[yellow_id + 1]
                yellow_pts = positions[st2:ed2]

                observed_state.append([red_pts, yellow_pts])

                p1.set_data(p_gt[current_fid, :n_particle], size=particle_size, edge_color='black', face_color=colors)
                line.set_data(pos=np.concatenate([p_gt[current_fid, :], floor_pos], axis=0), connect=gt_node_rs_idxs[current_fid])
                img = c.render()
                gt_out.write(img[:,:,:3])

            # model rollout
            data_path = os.path.join(trial_dir, f'{start_timestep}.h5')
            data = load_data_dominoes(data_names, data_path, phases_dict)
            data_path_prev = os.path.join(trial_dir, f'{int(start_timestep - args.training_fpt)}.h5')
            data_prev = load_data_dominoes(data_names, data_path_prev, phases_dict)
            _, data = recalculate_velocities([data_prev, data], dt, data_names)

            simulated_state = []
            node_rs_idxs = []
            for current_fid in range(max_timestep - start_id):
                p_pred[start_id + current_fid] = data[0]

                attr, state, rels, n_particles, n_shapes, instance_idx = \
                        prepare_input(data, args, phases_dict, args.verbose_data)

                Ra, node_r_idx, node_s_idx, pstep, rels_types = rels[3], rels[4], rels[5], rels[6], rels[7]
                node_rs_idxs.append(np.stack([rels[0][0][0], rels[1][0][0]], axis=1))

                Rr, Rs, Rr_idxs = [], [], []
                for j in range(len(rels[0])):
                    Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]
                    Rr_idxs.append(Rr_idx)
                    Rr.append(torch.sparse.FloatTensor(
                        Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))
                    Rs.append(torch.sparse.FloatTensor(
                        Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))

                buf = [attr, state, Rr, Rs, Ra, Rr_idxs]

                with torch.set_grad_enabled(False):
                    if use_gpu:
                        for d in range(len(buf)):
                            if type(buf[d]) == list:
                                for t in range(len(buf[d])):
                                    buf[d][t] = Variable(buf[d][t].cuda())
                            else:
                                buf[d] = Variable(buf[d].cuda())
                    else:
                        for d in range(len(buf)):
                            if type(buf[d]) == list:
                                for t in range(len(buf[d])):
                                    buf[d][t] = Variable(buf[d][t])
                            else:
                                buf[d] = Variable(buf[d])

                    attr, state, Rr, Rs, Ra, Rr_idxs = buf
                    vels = self.model(
                        attr, state, Rr, Rs, Ra, Rr_idxs, n_particles,
                        node_r_idx, node_s_idx, pstep, rels_types,
                        instance_idx, phases_dict, self.pretraining_cfg.TRAIN.args.verbose_model)

                vels = vels.cpu().numpy()
                data[0] = data[0] + (vels * dt)
                data[1][:, :args.position_dim] = vels

                positions = data[0]

                st, ed = instance_idx[red_id], instance_idx[red_id + 1]
                red_pts = positions[st:ed]

                st2, ed2 = instance_idx[yellow_id], instance_idx[yellow_id + 1]
                yellow_pts = positions[st2:ed2]

                simulated_state.append([red_pts, yellow_pts])

                p1.set_data(p_pred[start_id+current_fid, :n_particle], size=particle_size, edge_color='black', face_color=colors)
                line.set_data(pos=np.concatenate([p_pred[start_id+current_fid, :], floor_pos], axis=0), connect=node_rs_idxs[current_fid])
                img = c.render()
                pred_out.write(img[:,:,:3])

            input_states.append(input_state)
            observed_states.append(observed_state)
            simulated_states.append(simulated_state)

            gt_out.release()
            pred_out.release()
            mlflow.log_artifact(gt_vid_path, artifact_path='videos')
            mlflow.log_artifact(pred_vid_path, artifact_path='videos')
        output = {
            'input_states': input_states,
            'observed_states': observed_states,
            'simulated_states': simulated_states,
            'labels': labels,
            'stimulus_name': stimulus_name,
            }
        feature_file = os.path.join(self.output_dir, 'test_feat.pkl')
        pickle.dump(output, open(feature_file, 'wb'))
        logging.info('Saved features to {}'.format(feature_file))
        mlflow.log_artifact(feature_file, artifact_path=f'features')

class ReadoutObjective(ReadoutObjectiveBase):
    def get_readout_model(self):
        pass

    @staticmethod
    def get_thres(scenario):
        spacing = 0.05 # TODO: make arg
        if "Drape" in scenario:
            thres = 0.1222556027835 * 0.05/0.035
        elif "Contain" in scenario:
            thres = spacing * 1.0
        elif "Drop" in scenario:
            thres = spacing * 1.0
        else:
            thres = spacing * 1.5
        return thres

    def call(self, args):
        features = pickle.load(open(self.test_feature_file, 'rb'))
        thres = self.get_thres(self.readout_name)
        
        print(len(features['simulated_states']))
        accs = []
        for label, simulated_state in zip(features['labels'], features['simulated_states']):
            print(len(simulated_state))
            print(f'Yellow shape: {simulated_state[0][0].shape}')
            print(f'Red shape: {simulated_state[0][1].shape}')
            pred_is_positive_trial = False
            for yellow_pts, red_pts in simulated_state:
                sim_mat = scipy.spatial.distance_matrix(yellow_pts, red_pts, p=2)
                min_dist= np.min(sim_mat)

                pred_target_contacting_zone = min_dist < thres
                if pred_target_contacting_zone:
                    pred_is_positive_trial = True
                    break
            accs.append(pred_is_positive_trial==label)
        mlflow.log_metric('test_acc_simulated', np.mean(accs), step=self.restore_step)

def add_floor(v):
    # add floor
    floor_thickness = 0.025
    floor_length = 8.0
    w, h, d = floor_length, floor_length, floor_thickness
    b1 = vispy.scene.visuals.Box(width=w, height=h, depth=d, color=[0.8, 0.8, 0.8, 1], edge_color='black')
    #y_rotate(b1)
    v.add(b1)

    # adjust position of box
    mesh_b1 = b1.mesh.mesh_data
    v1 = mesh_b1.get_vertices()
    c1 = np.array([0., -floor_thickness*0.5, 0.], dtype=np.float32)
    mesh_b1.set_vertices(np.add(v1, c1))

    mesh_border_b1 = b1.border.mesh_data
    vv1 = mesh_border_b1.get_vertices()
    cc1 = np.array([0., -floor_thickness*0.5, 0.], dtype=np.float32)
    mesh_border_b1.set_vertices(np.add(vv1, cc1))

def create_instance_colors(n):
    # TODO: come up with a better way to initialize instance colors
    return np.array([
        [1., 1., 0., 1.],
        #[1., 0., 0., 1.],
        [0., 1., 0., 1.],
        [0., 0., 1., 1.],
        [1., 0., 1., 1.],
        [0., 1., 1., 1.],
        [1., 0., 0., 1.],
        [1., 1., 1., 1.],
        [1., 0.5, 0.5, 1.],
        [0.5, 0.5, 1., 1.],
        [0.5, 1., 0.5, 1.],
        [1., 0.25, 0.25, 1.],
        [0.25, 1., 0.25, 1.],
        [0.25, 0.25, 1., 1.],
        [0.25, 0.1, 1., 1.],
        [0.25, 0.1, 0.1, 1.],
        [0.1, 0.1, 1., 1.],])[:n]


def convert_groups_to_colors(group, instance_colors):
    """
    Convert grouping to RGB colors of shape (n_particles, 4)
    :param grouping: [p_rigid, p_instance, physics_param]
    :return: RGB values that can be set as color densities
    group: [0, 1024, 1032]
    """
    # p_rigid: n_instance
    # p_instance: n_p x n_instance
    n_instance = len(group) - 1
    n_particles = group[-1]

    #p_rigid, p_instance = group[:2]
    #p = p_instance

    colors = np.empty((n_particles, 4))

    for instance_id in range(n_instance):
        st, end = group[instance_id], group[instance_id+1]
        colors[st:end] = instance_colors[instance_id]

    colors = np.clip(colors, 0., 1.)
    return colors
