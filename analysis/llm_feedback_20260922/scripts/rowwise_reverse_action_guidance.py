#!/usr/bin/env python3
"""Row-wise x0 injection and branch-score-select action layer for TabDiff.

The mechanism borrows two patterns from visual diffusion work: correct the
predicted clean sample before the reverse update, and create multiple action
branches that are ranked by a frozen black-box reward.  Here the reward is a
typed graph certifier plus train-fitted kNN density and edit cost; no real row
or API call appears in the sampling loop.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from conditioned_tabdiff_adapter import decode, seed_all
from knowledge_graph_protocol import predicate_mask
from subgroup_projection_online import PreviewModel
from tabdiff_probe import preprocess_dataset
from typed_dynamic_graph_guidance import edge_masks, edge_metric_rows, graph_metrics


class RowActionLayer:
    def __init__(self, edges: list[dict], train_data, train: pd.DataFrame, info: dict,
                 density_weight: float, edit_weight: float, graph_weight: float=0.0,
                 action_margin_sd: float=0.0, action_planner: str="edge_local",
                 component_beam_width: int=4, component_violation_weight: float=100.0,
                 counterfactual_lookahead: bool=False, lookahead_margin: float=0.0,
                 lookahead_scope: str="target", lookahead_horizon: int=1,
                 component_min_edges: int=1, component_prototype_k: int=0,
                 calibration_policy: str="point", calibration_z: float=0.0,
                 lock_applicability_columns: bool=False,
                 preserve_satisfied_actuate: bool=False,
                 conditional_density_weight: float=0.0,
                 conditional_neighbors: int=64,
                 balanced_column_projection: bool=False,
                 edit_norm: str="l1",
                 balanced_min_t: float=0.0,
                 balanced_routing: str="fixed"):
        self.edges=edges; self.train_data=train_data; self.train=train
        self.density_weight=density_weight; self.edit_weight=edit_weight; self.graph_weight=graph_weight
        self.action_margin_sd=float(action_margin_sd)
        self.action_planner=str(action_planner)
        self.component_beam_width=int(component_beam_width)
        self.component_violation_weight=float(component_violation_weight)
        self.counterfactual_lookahead=bool(counterfactual_lookahead)
        self.lookahead_margin=float(lookahead_margin)
        self.lookahead_scope=str(lookahead_scope)
        self.lookahead_horizon=max(1,int(lookahead_horizon))
        self.component_min_edges=max(1,int(component_min_edges))
        self.component_prototype_k=max(0,int(component_prototype_k))
        self.prototype_feasible_cache={}
        self.calibration_policy=str(calibration_policy)
        self.calibration_z=max(0.0,float(calibration_z))
        # A user-specified implication must not be satisfied by changing its
        # antecedent.  Keep the legacy option for empirical dependency graphs,
        # where deactivation can be a legitimate counterfactual action.
        self.lock_applicability_columns=bool(lock_applicability_columns)
        # A satisfied, readable actuator is positive row-state evidence.  It
        # cannot trigger an edit, but it may veto a candidate action that would
        # turn the currently valid relation into a violation.  This closes the
        # previous asymmetry where only violations influenced action choice.
        self.preserve_satisfied_actuate=bool(preserve_satisfied_actuate)
        self.conditional_density_weight=max(0.0,float(conditional_density_weight))
        self.conditional_neighbors=max(4,int(conditional_neighbors))
        self.balanced_column_projection=bool(balanced_column_projection)
        self.edit_norm=str(edit_norm)
        self.balanced_min_t=float(balanced_min_t)
        self.balanced_routing=str(balanced_routing)
        self.balanced_switch_by_edge={}
        self.current_t=0.0
        self.conditional_neighbor_cache={}
        self.actuate_edges=[edge for edge in edges if edge.get("usage_role", "actuate")=="actuate"]
        self.score_edges=[edge for edge in edges if edge.get("usage_role")=="score_only"]
        self.active_score_edges=list(self.score_edges)
        self.active_actuate_guards=list(self.actuate_edges) if self.preserve_satisfied_actuate else []
        self.numeric_names=[info["column_names"][i] for i in info["num_col_idx"]]
        self.categorical_names=[info["column_names"][i] for i in info["target_col_idx"]+info["cat_col_idx"]]
        self.num_pos={name:i for i,name in enumerate(self.numeric_names)}
        self.cat_pos={name:i for i,name in enumerate(self.categorical_names)}
        self.scales={name:max(float(pd.to_numeric(train[name],errors="coerce").std()),1e-6) for name in self.numeric_names}
        matrix=train[self.numeric_names].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(float)
        self.scaler=StandardScaler().fit(matrix); scaled=self.scaler.transform(matrix)
        self.scaled_train_numeric=scaled
        self.knn=NearestNeighbors(n_neighbors=min(5,len(train))).fit(scaled)
        distances=self.knn.kneighbors(scaled[:min(2048,len(scaled))],return_distance=True)[0][:,-1]
        self.density_scale=max(float(np.median(distances)),1e-6)
        self.common_values={c:train[c].astype(str).value_counts().index.tolist() for c in self.categorical_names}

    def set_score_time(self, t: float, cutoffs: dict[str, float], schedule_mode: str,
                       common_cutoff: float) -> None:
        """Activate action guardrails only after their previews are readable.

        Applicability remains row-dependent inside ``score``.  This method adds
        the missing trajectory-dependent gate so a late-readable hyperedge
        cannot rank candidate actions at an earlier, noisy reverse step.
        """
        self.current_t=float(t)
        if schedule_mode == "edge":
            self.active_score_edges = [
                edge for edge in self.score_edges
                if t <= cutoffs.get(edge["candidate_id"], -math.inf)
            ]
        elif schedule_mode == "common":
            self.active_score_edges = list(self.score_edges) if t <= common_cutoff else []
        else:
            self.active_score_edges = list(self.score_edges)
        if self.preserve_satisfied_actuate:
            if schedule_mode == "edge":
                self.active_actuate_guards = [
                    edge for edge in self.actuate_edges
                    if t <= cutoffs.get(edge["candidate_id"], -math.inf)
                ]
            elif schedule_mode == "common":
                self.active_actuate_guards = list(self.actuate_edges) if t <= common_cutoff else []
            else:
                self.active_actuate_guards = list(self.actuate_edges)

    def numeric_target(self, column: str, operator: str, value: float, satisfy: bool=True) -> float:
        raw=pd.to_numeric(self.train[column],errors="coerce").dropna().to_numpy(float)
        if not satisfy:
            inverse={"==":"!=","!=":"==",">":"<=",">=":"<","<":">=","<=":">"}[operator]
            return self.numeric_target(column,inverse,value,True)
        if operator=="==": return float(value)
        if operator=="!=":
            other=raw[~np.isclose(raw,float(value))]; return float(np.median(other)) if len(other) else float(value)+1
        margin=self.action_margin_sd*self.scales[column]
        if operator==">":
            other=raw[raw>float(value)+margin]; return float(other.min()) if len(other) else float(value)+max(margin,np.finfo(float).eps)
        if operator==">=": return float(value)+margin
        if operator=="<":
            other=raw[raw<float(value)-margin]; return float(other.max()) if len(other) else float(value)-max(margin,np.finfo(float).eps)
        if operator=="<=": return float(value)-margin
        raise ValueError(operator)

    def apply_column_relation(self, row: pd.DataFrame, atom: dict, column_side: str) -> None:
        """Move a column relation into its feasible interior rather than its boundary.

        Boundary projection is brittle during reverse diffusion: the next model
        update can flip an equality-valued inequality back to violation.  A small
        scale-normalized margin makes the action persistent without hard-clamping
        the remaining trajectory.
        """
        left,right=atom["left_column"],atom["right_column"]
        operator=atom["operator"]
        left_value=float(row.iloc[0][left]); right_value=float(row.iloc[0][right])
        if column_side=="right":
            margin=self.action_margin_sd*self.scales.get(right,1.0)
            targets={"==":left_value,"<=":left_value+margin,"<":left_value+max(margin,np.finfo(float).eps),
                     ">=":left_value-margin,">":left_value-max(margin,np.finfo(float).eps),
                     "!=":left_value+max(margin,.05*self.scales.get(right,1.0),np.finfo(float).eps)}
            if operator in targets: row.loc[row.index[0],right]=targets[operator]
        else:
            margin=self.action_margin_sd*self.scales.get(left,1.0)
            targets={"==":right_value,"<=":right_value-margin,"<":right_value-max(margin,np.finfo(float).eps),
                     ">=":right_value+margin,">":right_value+max(margin,np.finfo(float).eps),
                     "!=":right_value+max(margin,.05*self.scales.get(left,1.0),np.finfo(float).eps)}
            if operator in targets: row.loc[row.index[0],left]=targets[operator]

    def apply_balanced_column_projection(self, row: pd.DataFrame, atom: dict) -> None:
        """Weighted nearest-point projection for a two-column numeric relation.

        Editing only one endpoint is often unnecessarily destructive.  This
        projection minimizes the squared standardized displacement of both
        endpoints under the requested equality or inequality.  The ordinary
        reward layer still decides whether this candidate is preferable to a
        one-sided edit or abstention.
        """
        left, right = atom["left_column"], atom["right_column"]
        if left not in self.num_pos or right not in self.num_pos:
            return
        operator = atom["operator"]
        if operator == "!=":
            return
        left_value = float(row.iloc[0][left]); right_value = float(row.iloc[0][right])
        tolerance = float(atom.get("tolerance", 0.0) or 0.0)
        # Use an averaged scale for the requested interior margin, then divide
        # the required correction according to standardized edit cost.
        left_scale = self.scales.get(left, 1.0); right_scale = self.scales.get(right, 1.0)
        margin = self.action_margin_sd * .5 * (left_scale + right_scale)
        difference = left_value - right_value
        # ``tolerance`` is a verifier allowance, not the semantic boundary.
        # Project to x=y (or slightly inside for a strict inequality) so a
        # decode/serialization round trip cannot move a boundary-valued action
        # back outside the certified feasible set.
        strict_interior = max(np.finfo(float).eps * max(left_scale, right_scale), .01 * tolerance)
        if operator in {"<=", "<"}:
            target_difference = -max(margin, strict_interior if operator == "<" else 0.0)
        elif operator in {">=", ">"}:
            target_difference = max(margin, strict_interior if operator == ">" else 0.0)
        else:  # equality
            if abs(difference) <= tolerance:
                return
            target_difference = 0.0
        correction = target_difference - difference
        denominator = left_scale ** 2 + right_scale ** 2
        if denominator <= 0:
            return
        left_delta = correction * left_scale ** 2 / denominator
        right_delta = -correction * right_scale ** 2 / denominator
        row.loc[row.index[0], left] = left_value + left_delta
        row.loc[row.index[0], right] = right_value + right_delta

    def apply_literal(self, row: pd.DataFrame, atom: dict, satisfy: bool=True) -> None:
        column=atom["left_column"]; operator=atom["operator"]; value=atom["right_value"]
        if column in self.num_pos:
            row.loc[row.index[0],column]=self.numeric_target(column,operator,float(value),satisfy)
        else:
            if satisfy:
                if operator=="==": row.loc[row.index[0],column]=str(value)
                elif operator=="!=":
                    choices=[x for x in self.common_values[column] if x!=str(value)]
                    if choices: row.loc[row.index[0],column]=choices[0]
            else:
                if operator=="==":
                    choices=[x for x in self.common_values[column] if x!=str(value)]
                    if choices: row.loc[row.index[0],column]=choices[0]
                elif operator=="!=": row.loc[row.index[0],column]=str(value)

    def direct_candidate(self, row: pd.DataFrame, edge: dict, column_side: str="left") -> pd.DataFrame:
        out=row.copy()
        for atom in edge.get("requirements",[]):
            if bool(predicate_mask(out,atom)[0]): continue
            if atom.get("right_kind")=="literal": self.apply_literal(out,atom,True)
            else: self.apply_column_relation(out,atom,column_side)
        return out

    def balanced_candidate(self, row: pd.DataFrame, edge: dict) -> pd.DataFrame:
        out = row.copy()
        for atom in edge.get("requirements", []):
            if bool(predicate_mask(out, atom)[0]):
                continue
            if atom.get("right_kind") == "column":
                self.apply_balanced_column_projection(out, atom)
            else:
                self.apply_literal(out, atom, True)
        return out

    def balanced_allowed(self, edge: dict) -> bool:
        """Route action geometry using train-only trajectory readability.

        Reverse time decreases from one to zero.  Before the clean preview is
        reliable enough for a stable endpoint decision, offer the symmetric
        minimum-displacement projection.  Once the edge-specific violation
        state reaches the calibrated precision/recall threshold, use the more
        persistent one-sided endpoints.  ``fixed`` preserves the pre-existing
        global-cutoff ablation.
        """
        threshold=self.balanced_min_t
        if self.balanced_routing=="trajectory_reliability":
            threshold=float(self.balanced_switch_by_edge.get(edge["candidate_id"],threshold))
        return self.current_t >= threshold

    def candidates(self, row: pd.DataFrame, edge: dict) -> list[tuple[str,pd.DataFrame]]:
        result=[("direct_requirements",self.direct_candidate(row,edge,"left"))]
        if any(atom.get("right_kind")=="column" for atom in edge.get("requirements",[])):
            result.append(("alternate_endpoint",self.direct_candidate(row,edge,"right")))
            if self.balanced_column_projection and self.balanced_allowed(edge):
                result.append(("balanced_projection", self.balanced_candidate(row, edge)))
        if not self.lock_applicability_columns:
            for index,atom in enumerate(edge.get("applicability",[])):
                alternative=row.copy()
                if atom.get("right_kind")=="literal": self.apply_literal(alternative,atom,False)
                else:
                    left,right=atom["left_column"],atom["right_column"]
                    value=float(alternative.iloc[0][right]); alternative.loc[alternative.index[0],left]=value+max(self.scales.get(left,1.0)*.05,1e-6)
                result.append((f"deactivate_{index}",alternative))
        result.append(("abstain",row.copy()))
        dedup={}
        for label,item in result:
            key=tuple(str(item.iloc[0][c]) for c in item.columns)
            dedup.setdefault(key,(label,item))
        return list(dedup.values())

    def changed_columns(self, original: pd.DataFrame, candidate: pd.DataFrame) -> list[str]:
        changed=[]
        for column in original.columns:
            a,b=original.iloc[0][column],candidate.iloc[0][column]
            try:
                # ``np.isclose`` defaults to rtol=1e-5.  That is unsafe for
                # monetary/count columns: at magnitude 5,000 it silently drops
                # one side of a 0.03 balanced projection, destroying the joint
                # feasibility guarantee.  Ignore only numerical round-off, not
                # scale-relative action deltas.
                left=float(a); right=float(b)
                atol=8.0*np.finfo(np.float32).eps*max(1.0,abs(left),abs(right))
                equal=bool(np.isclose(left,right,rtol=0.0,atol=atol,equal_nan=True))
            except Exception: equal=str(a)==str(b)
            if not equal: changed.append(column)
        return changed

    def conditional_density(self, original: pd.DataFrame, candidate: pd.DataFrame,
                            targets: tuple[str,...]) -> float:
        """Train-only local plausibility of edited cells given untouched context.

        This reuses the successful conditional-density idea from the repainting
        branch inside the trajectory action selector.  It never changes graph
        activation and never consults held-out rows.
        """
        if not targets or self.conditional_density_weight<=0:
            return 0.0
        key=tuple(sorted(targets))
        if key not in self.conditional_neighbor_cache:
            positions=[i for i,name in enumerate(self.numeric_names) if name not in key]
            if not positions: positions=list(range(len(self.numeric_names)))
            model=NearestNeighbors(n_neighbors=min(self.conditional_neighbors,len(self.train))).fit(
                self.scaled_train_numeric[:,positions])
            self.conditional_neighbor_cache[key]=(positions,model)
        positions,model=self.conditional_neighbor_cache[key]
        raw=original[self.numeric_names].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(float)
        encoded=self.scaler.transform(raw)
        neighbours=model.kneighbors(encoded[:,positions],return_distance=False)[0]
        terms=[]
        for target in key:
            if target in self.numeric_names:
                bank=pd.to_numeric(self.train.iloc[neighbours][target],errors="coerce").dropna().to_numpy(float)
                if not len(bank): continue
                value=float(pd.to_numeric(candidate[target],errors="coerce").iloc[0])
                global_scale=self.scales[target]
                q25,q75=np.quantile(bank,[.25,.75])
                local_scale=max(float((q75-q25)/1.349),.05*global_scale)
                nearest=float(np.min(np.abs(bank-value))/global_scale)
                central=float(abs(value-np.median(bank))/local_scale)
                terms.append(min(20.0,.5*nearest+.5*central))
            elif target in self.categorical_names:
                bank=self.train.iloc[neighbours][target].astype(str)
                value=str(candidate.iloc[0][target])
                cardinality=max(len(self.common_values[target]),1)
                probability=(int(bank.eq(value).sum())+1.0)/(len(bank)+cardinality)
                terms.append(float(-math.log(max(probability,1e-8))))
        return float(np.mean(terms)) if terms else 0.0

    def edit_distance(self, original: pd.DataFrame, candidate: pd.DataFrame) -> float:
        numeric = [
            (float(candidate.iloc[0][column]) - float(original.iloc[0][column])) / self.scales[column]
            for column in self.numeric_names
        ]
        categorical = [
            float(str(candidate.iloc[0][column]) != str(original.iloc[0][column]))
            for column in self.categorical_names
        ]
        values = numeric + categorical
        if self.edit_norm == "l2":
            return float(math.sqrt(sum(value * value for value in values)))
        return float(sum(abs(value) for value in values))

    def score(self, original: pd.DataFrame, candidate: pd.DataFrame, edge: dict, label: str) -> float:
        bad=bool(edge_masks(candidate,edge)[2][0])
        values=candidate[self.numeric_names].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(float)
        density=float(self.knn.kneighbors(self.scaler.transform(values),return_distance=True)[0][0,-1]/self.density_scale)
        edit=self.edit_distance(original,candidate)
        targets=tuple(self.changed_columns(original,candidate))
        conditional=self.conditional_density(original,candidate,targets)
        # Score-only edges are guardrails, not extra hard constraints. Penalize
        # only violations introduced by this candidate; do not force repair of
        # pre-existing empirical exceptions.
        graph_penalty=0.0
        if self.graph_weight>0:
            class_weight={"hard":3.0,"near_hard":2.0,"soft":1.0}
            for guard in self.active_score_edges + self.active_actuate_guards:
                original_bad=bool(edge_masks(original,guard)[2][0])
                candidate_bad=bool(edge_masks(candidate,guard)[2][0])
                if candidate_bad and not original_bad:
                    graph_penalty+=class_weight.get(guard.get("classification"),1.0)
        direction_penalty=.20 if label.startswith("deactivate") else 0.0
        abstain_penalty=.05 if label=="abstain" else 0.0
        return (100.0*float(bad)+self.density_weight*density+self.edit_weight*edit+
                self.conditional_density_weight*conditional+
                self.graph_weight*graph_penalty+direction_penalty+abstain_penalty)

    @staticmethod
    def edge_columns(edge: dict) -> set[str]:
        columns={atom["left_column"] for atom in edge.get("applicability",[])+edge.get("requirements",[])}
        columns|={atom["right_column"] for atom in edge.get("applicability",[])+edge.get("requirements",[])
                  if atom.get("right_kind")=="column"}
        return columns

    def edge_components(self, edges: list[dict]) -> list[list[dict]]:
        """Connected components induced by columns touched by gated edges."""
        remaining=list(edges); components=[]
        while remaining:
            component=[remaining.pop(0)]; columns=set(self.edge_columns(component[0]))
            changed=True
            while changed:
                changed=False
                keep=[]
                for edge in remaining:
                    edge_columns=self.edge_columns(edge)
                    if columns.intersection(edge_columns):
                        component.append(edge); columns|=edge_columns; changed=True
                    else:
                        keep.append(edge)
                remaining=keep
            components.append(component)
        return components

    def joint_score(self, original: pd.DataFrame, candidate: pd.DataFrame,
                    target_edges: list[dict], labels: tuple[str,...]) -> float:
        """Energy for one row-level counterfactual over an active subgraph.

        Unlike the edge-local reward, this score treats simultaneously gated
        constraints as one factor graph.  It also rejects collateral violations
        introduced on other discovered edges, while not forcing repair of their
        pre-existing empirical exceptions.
        """
        class_weight={"hard":3.0,"near_hard":2.0,"soft":1.0}
        target_ids={edge["candidate_id"] for edge in target_edges}
        violation_energy=0.0
        for edge in target_edges:
            if bool(edge_masks(candidate,edge)[2][0]):
                violation_energy+=class_weight.get(edge.get("classification"),1.0)

        collateral=0.0
        guard_edges=self.actuate_edges+self.active_score_edges
        for edge in guard_edges:
            if edge["candidate_id"] in target_ids: continue
            original_bad=bool(edge_masks(original,edge)[2][0])
            candidate_bad=bool(edge_masks(candidate,edge)[2][0])
            if candidate_bad and not original_bad:
                collateral+=class_weight.get(edge.get("classification"),1.0)

        values=candidate[self.numeric_names].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(float)
        density=float(self.knn.kneighbors(self.scaler.transform(values),return_distance=True)[0][0,-1]/self.density_scale)
        edit=self.edit_distance(original,candidate)
        direction_penalty=.20*sum(label.startswith("deactivate") for label in labels)
        abstain_penalty=.05*sum(label=="abstain" for label in labels)
        return (self.component_violation_weight*violation_energy+
                self.graph_weight*collateral+self.density_weight*density+
                self.edit_weight*edit+direction_penalty+abstain_penalty)

    def conditional_prototype(self, row: pd.DataFrame, component: list[dict]) -> pd.DataFrame | None:
        """Train-only local prototype that satisfies an active component.

        Neighbours are selected on columns outside the component, then the
        involved values are aggregated by median/mode.  This gives the action
        planner a manifold-aware alternative without copying an entire real
        record or exposing real rows to an external model.
        """
        if self.component_prototype_k<=0:
            return None
        involved=set().union(*(self.edge_columns(edge) for edge in component))
        component_key=tuple(sorted(edge["candidate_id"] for edge in component))
        indices=self.prototype_feasible_cache.get(component_key)
        if indices is None:
            feasible=np.ones(len(self.train),dtype=bool)
            for edge in component:
                active,satisfied,_=edge_masks(self.train,edge)
                feasible &= active & satisfied
            indices=np.flatnonzero(feasible)
            self.prototype_feasible_cache[component_key]=indices
        if not len(indices):
            return None

        distances=np.zeros(len(indices),dtype=float)
        for column in self.numeric_names:
            if column in involved: continue
            values=pd.to_numeric(self.train.iloc[indices][column],errors="coerce").fillna(0).to_numpy(float)
            target=float(pd.to_numeric(row[column],errors="coerce").fillna(0).iloc[0])
            distances+=((values-target)/self.scales[column])**2
        for column in self.categorical_names:
            if column in involved: continue
            values=self.train.iloc[indices][column].astype(str).to_numpy()
            distances+=(values!=str(row.iloc[0][column])).astype(float)
        neighbours=indices[np.argsort(distances)[:min(self.component_prototype_k,len(indices))]]
        candidate=row.copy()
        for column in involved:
            values=self.train.iloc[neighbours][column]
            if column in self.numeric_names:
                candidate.loc[candidate.index[0],column]=float(pd.to_numeric(values,errors="coerce").median())
            elif column in self.categorical_names:
                modes=values.mode(dropna=False)
                if len(modes): candidate.loc[candidate.index[0],column]=modes.iloc[0]
        if any(bool(edge_masks(candidate,edge)[2][0]) for edge in component):
            return None
        return candidate

    def choose_component(self, row: pd.DataFrame, target_edges: list[dict]) -> tuple[pd.DataFrame,dict[str,str],list[str],float,int]:
        """Jointly plan edits inside each connected active constraint component."""
        result=row.copy(); selected_labels={}; total_score=0.0; changed_components=0
        ordered=sorted(target_edges,key=lambda edge:(
            {"hard":0,"near_hard":1,"soft":2}.get(edge.get("classification"),3),edge["candidate_id"]))
        for component in self.edge_components(ordered):
            component_original=result.copy()
            beams=[(component_original.copy(),tuple(),{})]
            for edge in component:
                expanded=[]
                for candidate_row,labels,label_map in beams:
                    for label,next_row in self.candidates(candidate_row,edge):
                        next_labels=labels+(label,)
                        next_map={**label_map,edge["candidate_id"]:label}
                        score=self.joint_score(component_original,next_row,component,next_labels)
                        expanded.append((score,next_row,next_labels,next_map))
                dedup={}
                for score,next_row,next_labels,next_map in sorted(expanded,key=lambda item:item[0]):
                    key=tuple(str(next_row.iloc[0][column]) for column in next_row.columns)
                    if key not in dedup:
                        dedup[key]=(next_row,next_labels,next_map,score)
                beams=[(item[0],item[1],item[2]) for item in list(dedup.values())[:self.component_beam_width]]
            ranked=[(self.joint_score(component_original,candidate,component,labels),candidate,labels,label_map)
                    for candidate,labels,label_map in beams]
            prototype=self.conditional_prototype(component_original,component)
            if prototype is not None:
                labels=tuple("conditional_prototype" for _ in component)
                label_map={edge["candidate_id"]:"conditional_prototype" for edge in component}
                ranked.append((self.joint_score(component_original,prototype,component,labels),
                               prototype,labels,label_map))
            score,best,labels,label_map=min(ranked,key=lambda item:item[0])
            if self.changed_columns(component_original,best): changed_components+=1
            result=best; selected_labels.update(label_map); total_score+=float(score)
        return result,selected_labels,self.changed_columns(row,result),total_score,changed_components

    def choose(self, row: pd.DataFrame, edge: dict, mode: str,
               require_feasible: bool=False) -> tuple[str,pd.DataFrame,list[str],float]:
        if mode=="directed":
            candidate=self.direct_candidate(row,edge)
            return "direct_requirements",candidate,self.changed_columns(row,candidate),0.0
        scored=[]
        for label,candidate in self.candidates(row,edge):
            if require_feasible and bool(edge_masks(candidate,edge)[2][0]):
                continue
            scored.append((self.score(row,candidate,edge,label),label,candidate))
        if not scored:
            candidate=self.direct_candidate(row,edge)
            return "direct_requirements",candidate,self.changed_columns(row,candidate),0.0
        score,label,candidate=min(scored,key=lambda x:x[0])
        return label,candidate,self.changed_columns(row,candidate),float(score)

    def encode_numeric(self, frame: pd.DataFrame) -> np.ndarray:
        matrix=frame[self.numeric_names].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(float)
        return self.train_data.num_transform.transform(matrix)

    def encode_category(self, column: str, value) -> int:
        categories=self.train_data.cat_transform.named_steps["ordinal"].categories_[self.cat_pos[column]]
        hit=np.flatnonzero(np.asarray(categories).astype(str)==str(value))
        if len(hit)!=1: raise ValueError(f"unknown category {column}={value}")
        return int(hit[0])

    def severity(self, frame: pd.DataFrame, edge: dict) -> np.ndarray:
        total=np.zeros(len(frame),dtype=float)
        for atom in edge.get("requirements",[]):
            bad=~predicate_mask(frame,atom)
            left=pd.to_numeric(frame[atom["left_column"]],errors="coerce").to_numpy(float) if atom["left_column"] in self.num_pos else np.zeros(len(frame))
            if atom.get("right_kind")=="column":
                right=pd.to_numeric(frame[atom["right_column"]],errors="coerce").to_numpy(float)
                total+=bad*np.abs(left-right)/self.scales[atom["left_column"]]
            elif atom["left_column"] in self.num_pos:
                total+=bad*np.abs(left-float(atom["right_value"]))/self.scales[atom["left_column"]]
            else: total+=bad.astype(float)
        return total


def effective_calibration_rate(edge: dict, layer: RowActionLayer) -> float:
    """Return a train-only exception target for probabilistic constraints."""
    calibration=edge["calibration"]
    rate=float(calibration["violation_rate"])
    split_names=("discovery","counterexample","calibration")
    rates=[float(edge[name]["violation_rate"]) for name in split_names
           if name in edge and np.isfinite(float(edge[name].get("violation_rate",math.nan)))]
    if layer.calibration_policy in {"split_median","split_upper","split_wilson_upper"}:
        if rates:
            rate=float(np.median(rates))
            if layer.calibration_policy=="split_upper":
                rate=float(np.max(rates))
    if layer.calibration_policy in {"wilson_upper","split_wilson_upper"} and layer.calibration_z>0:
        if layer.calibration_policy=="split_wilson_upper":
            supports=[int(edge[name].get("support",0)) for name in split_names if name in edge]
            n=max(1,int(np.median(supports))) if supports else max(1,int(calibration.get("support",0)))
            successes=rate*n
        else:
            n=max(1,int(calibration.get("support",0)))
            successes=float(calibration.get("violations",rate*n))
        p=min(1.0,max(0.0,successes/n)); z=layer.calibration_z
        denominator=1.0+z*z/n
        centre=p+z*z/(2.0*n)
        radius=z*math.sqrt((p*(1.0-p)+z*z/(4.0*n))/n)
        rate=min(1.0,(centre+radius)/denominator)
    return rate


def graph_rate_candidates(frame: pd.DataFrame, edge: dict, indices: np.ndarray,
                          excess: int, layer: RowActionLayer) -> np.ndarray:
    """Choose actions by their joint effect on the train-calibrated graph.

    A local repair can activate a downstream edge or change its denominator.
    Ranking only by the currently violated edge therefore treats a conditional
    graph as independent rules.  This batch-level controller evaluates each
    feasible row action against the complete vector of conditional violation
    rates and greedily selects the subset nearest to the train-only targets.
    It never reads held-out rows or held-out rates.
    """
    graph_edges=list(layer.actuate_edges)
    if not graph_edges or excess<=0 or not len(indices):
        return np.asarray([],dtype=int)
    active_counts=[]; violation_counts=[]; targets=[]
    for graph_edge in graph_edges:
        active,_,violation=edge_masks(frame,graph_edge)
        active_counts.append(int(active.sum()))
        violation_counts.append(int(violation.sum()))
        targets.append(effective_calibration_rate(graph_edge,layer))
    active_counts=np.asarray(active_counts,dtype=float)
    violation_counts=np.asarray(violation_counts,dtype=float)
    targets=np.asarray(targets,dtype=float)
    delta_active=np.zeros((len(indices),len(graph_edges)),dtype=float)
    delta_violation=np.zeros_like(delta_active)
    feasible=np.ones(len(indices),dtype=bool)
    local_scores=np.zeros(len(indices),dtype=float)
    for position,index in enumerate(indices):
        original=frame.iloc[[int(index)]]
        _,candidate,columns,score=layer.choose(original,edge,"reward")
        if not columns:
            feasible[position]=False
            local_scores[position]=math.inf
            continue
        local_scores[position]=float(score)
        for column,graph_edge in enumerate(graph_edges):
            before_active,_,before_violation=edge_masks(original,graph_edge)
            after_active,_,after_violation=edge_masks(candidate,graph_edge)
            delta_active[position,column]=float(after_active[0])-float(before_active[0])
            delta_violation[position,column]=float(after_violation[0])-float(before_violation[0])
    selected=[]
    remaining=feasible.copy()
    for _ in range(min(int(excess),int(feasible.sum()))):
        candidates=np.flatnonzero(remaining)
        if not len(candidates):
            break
        next_active=np.maximum(1.0,active_counts[None,:]+delta_active[candidates])
        next_violation=np.maximum(0.0,violation_counts[None,:]+delta_violation[candidates])
        next_rates=next_violation/next_active
        objectives=np.mean((next_rates-targets[None,:])**2,axis=1)
        # The density/edit reward is used only as a deterministic tie-breaker;
        # graph-rate calibration remains the primary objective.
        objectives=objectives+1e-12*local_scores[candidates]
        best_position=int(candidates[int(np.argmin(objectives))])
        selected.append(int(indices[best_position]))
        active_counts+=delta_active[best_position]
        violation_counts+=delta_violation[best_position]
        remaining[best_position]=False
    return np.asarray(selected,dtype=int)


def calibrated_gate(frame: pd.DataFrame, edge: dict, gate: np.ndarray, layer: RowActionLayer,
                    trajectory_precision: float=1.0, trajectory_recall: float=1.0,
                    gate_ranking: str="severity", row_posterior: np.ndarray | None=None) -> np.ndarray:
    if edge["classification"]=="hard" or not gate.any(): return gate
    active,_,violation=edge_masks(frame,edge)
    rate=effective_calibration_rate(edge,layer)
    allowed=int(round(rate*int(active.sum())))
    # A middle-state violation is not necessarily a final violation. Estimate
    # the final mass from trajectory precision/recall before spending action
    # authority, so transient errors do not erase valid train exceptions.
    precision=float(np.clip(trajectory_precision,0.0,1.0))
    recall=float(np.clip(trajectory_recall,1e-6,1.0))
    predicted_final=int(round(int(violation.sum())*precision/recall))
    excess=max(0,predicted_final-allowed); indices=np.flatnonzero(gate)
    if excess>=len(indices): return gate
    if gate_ranking=="graph_rate":
        chosen=graph_rate_candidates(frame,edge,indices,excess,layer)
    elif gate_ranking in {"trajectory_posterior","learned_posterior"} and row_posterior is not None:
        chosen=indices[np.argsort(-row_posterior[indices])[:excess]]
    elif gate_ranking=="posterior_cost":
        # Parameter-free Borda fusion of two independently meaningful action
        # criteria: probability that the visible violation survives to x0,
        # and train-density/edit/collateral cost of the feasible correction.
        # For edges without a learned certifier this reduces to action cost.
        costs=[]
        for index in indices:
            _,_,columns,score=layer.choose(frame.iloc[[int(index)]],edge,"reward")
            costs.append(float(score) if columns else math.inf)
        costs=np.asarray(costs,dtype=float)
        if row_posterior is None:
            chosen=indices[np.argsort(costs)[:excess]]
        else:
            n=max(1,len(indices)-1)
            probability_rank=np.argsort(np.argsort(row_posterior[indices])).astype(float)/n
            low_cost_rank=1.0-np.argsort(np.argsort(costs)).astype(float)/n
            utility=probability_rank+low_cost_rank
            chosen=indices[np.argsort(-utility)[:excess]]
    elif gate_ranking=="action_cost":
        # Near-hard rules define a batch-level violation allowance.  Preserve
        # plausible high-cost exceptions and spend the correction quota on the
        # rows with the cheapest density/edit/collateral-safe feasible action.
        costs=[]
        for index in indices:
            _,_,columns,score=layer.choose(frame.iloc[[int(index)]],edge,"reward")
            costs.append(float(score) if columns else math.inf)
        chosen=indices[np.argsort(np.asarray(costs))[:excess]]
    else:
        chosen=indices[np.argsort(-layer.severity(frame,edge)[indices])[:excess]]
    result=np.zeros_like(gate); result[chosen]=True; return result


def role_metrics(frame: pd.DataFrame, reference: pd.DataFrame, edges: list[dict]) -> dict:
    result={}
    for role in ("actuate", "score_only", "monitor"):
        subset=[edge for edge in edges if edge.get("usage_role", "actuate")==role]
        if not subset: continue
        for key,value in graph_metrics(frame,reference,subset).items():
            result[f"{role}_{key}"]=value
    return result


@torch.no_grad()
def sample_guided(diffusion,preview,train_data,info,edges,layer,cutoffs,samples,mode,min_support,drift,
                  max_actions_per_row_edge=0,cooldown_steps=0,schedule_mode="edge",gate_mode="persistent",
                  common_cutoff=.306,topology_mode="dynamic",active_min_t=.10,active_max_t=.75,
                  topology_refresh_interval=1,state_gate="current",oracle_masks=None,max_total_actions=0,
                  topology_persistence=2,volatility_threshold=.0015,volatility_ema=.8,
                  numeric_guidance_mode="constant",numeric_guidance_scale=1.0,
                  target_step_fraction=.10,max_guidance_scale=8.0,action_hold_steps=0,
                  hard_guidance_scale=4.0,near_guidance_scale=1.0,soft_guidance_scale=.5,
                  feedback_repeat_growth=1.5,feedback_late_multiplier=2.0,
                  max_guidance_delta=0.0,promote_score_only=False,
                  promoted_min_streak=3,promoted_max_calibration_rate=.05,
                  trajectory_profiles=None,xstate_blend=1.0,gate_ranking="severity",
                  streak_profiles=None,trajectory_certifiers=None,
                  dual_lr=.5,dual_decay=.9,dual_max=8.0,dual_threshold=1.0):
    device=diffusion.device; started=time.perf_counter()
    times=torch.linspace(0,1,diffusion.num_timesteps,dtype=torch.float32,device=device)[:,None]
    sn=diffusion.num_schedule.total_noise(times); sc=diffusion.cat_schedule.total_noise(times)
    snx=torch.zeros_like(sn); snx[1:]=sn[:-1]; scx=torch.zeros_like(sc); scx[1:]=sc[:-1]
    numeric=torch.randn((samples,diffusion.num_numerical_features),device=device)*sn[-1]
    categorical=diffusion._sample_masked_prior(samples,len(diffusion.num_classes))
    previous_masks={}; frozen_masks={}; frozen_violations={}; topology_previous={}; violation_previous={}
    topology_raw_previous={}; topology_volatility={}; topology_row_volatility={}; adaptive_persistence_counts=Counter()
    violation_streaks={}
    dual_state={edge["candidate_id"]:np.zeros(samples,dtype=np.float32) for edge in edges}
    action_holds={}
    last_topology_refresh={}; topology_mismatch_streaks={}; previous_numeric=None
    topology_changed_rows=np.zeros(samples,dtype=bool); topology_changes_by_edge=Counter()
    constraint_state_changed_rows=np.zeros(samples,dtype=bool); constraint_state_changes_by_edge=Counter()
    ever=np.zeros(samples,dtype=bool); accepted_ever=np.zeros(samples,dtype=bool); choices=Counter()
    action_counts={edge["candidate_id"]:np.zeros(samples,dtype=np.int16) for edge in edges}
    last_action={edge["candidate_id"]:np.full(samples,-10000,dtype=np.int16) for edge in edges}
    edge_by_id={edge["candidate_id"]:edge for edge in edges}
    stats={"action_events":0,"unique_rows_actuated":0,"conflicts":0,"actions_by_edge":{},"action_choices":{},
           "promoted_action_events":0,"component_action_events":0,"resolved_edge_events":0,
           "structural_deferred_edge_events":0,"structural_deferred_rows":0,
           "changed_cells":0,"attempted_action_events":0,"lookahead_accepted_events":0,
           "lookahead_rejected_events":0,"lookahead_accepted_rows":0,"lookahead_rejected_rows":0,
           "trace":[]}
    realized_guidance_scales=[]; guidance_clipped_cells=0
    rollback_mode=mode.startswith("rollback")
    action_mode="reward" if "reward" in mode or rollback_mode else "directed"
    x0_injection=mode.startswith("x0")
    rollback_generator=torch.Generator(device=device); rollback_generator.manual_seed(2026100607)
    for reverse_step,index in enumerate(reversed(range(diffusion.num_timesteps))):
        t=times[index]; sigma=sn[index]; sigma_next=snx[index]
        layer.set_score_time(float(t), cutoffs, schedule_mode, common_cutoff)
        # Mirror UnifiedCtimeDiffusion.edm_update exactly, including zero-size
        # churn calls that still advance the RNG. This makes paired-noise drift
        # meaningful against diffusion.sample under deterministic settings.
        numeric_hat=numeric+(sigma**2-sigma**2).sqrt()*torch.randn_like(numeric)
        move_chance=-torch.expm1(sc[index]-sc[index])
        categorical_hat,_=diffusion.q_xt(categorical,move_chance)
        xoh=diffusion.to_one_hot(categorical_hat).to(numeric.dtype)
        denoised,raw_logits=preview.base(numeric_hat.float(),xoh,t.squeeze().repeat(samples),sigma=sigma.unsqueeze(0).repeat(samples,1))
        codes=preview.predicted_codes(xoh,raw_logits)
        frame=decode(train_data,torch.cat([denoised,codes.float()],dim=1),info)
        corrected=frame.copy(); changed_numeric=np.zeros((samples,len(layer.numeric_names)),dtype=bool)
        changed_cat=np.zeros((samples,len(layer.categorical_names)),dtype=bool); claimed=[set() for _ in range(samples)]
        lookahead_targets=[[] for _ in range(samples)]; lookahead_labels=[[] for _ in range(samples)]
        lookahead_event_counts=np.zeros(samples,dtype=np.int16)
        cell_guidance_scale=np.ones((samples,len(layer.numeric_names)),dtype=np.float32)
        current_active_by_edge={}
        step={"t":float(t),"actions":0,"edges":0,"active_row_edges":0,"hold_cell_applications":0,
              "gate_required_streak":0,
              "topology_additions":0,"topology_removals":0,"topology_flips":0,"topology_refreshes":0,
              "dual_mass":0.0,"dual_nonzero":0,"dual_max":0.0}
        if active_min_t<=float(t)<=active_max_t:
            current_numeric=denoised.detach().cpu().numpy()
            pending_component_gates=[]
            for edge in edges:
                role=edge.get("usage_role", "actuate")
                calibration_rate=float(edge.get("calibration",{}).get("violation_rate",1.0))
                promoted=(promote_score_only and role=="score_only" and
                          calibration_rate<=promoted_max_calibration_rate)
                if role!="actuate" and not promoted: continue
                edge_id=edge["candidate_id"]
                if schedule_mode=="edge" and float(t)>cutoffs.get(edge_id,.25): continue
                if schedule_mode=="common" and float(t)>common_cutoff: continue
                raw_active,satisfied,raw_violation=edge_masks(frame,edge)
                if topology_mode=="frozen":
                    if edge_id not in frozen_masks:
                        frozen_masks[edge_id]=raw_active.copy(); step["topology_refreshes"]+=1
                    active=frozen_masks[edge_id].copy()
                elif topology_mode=="frozen_state":
                    if edge_id not in frozen_masks:
                        frozen_masks[edge_id]=raw_active.copy(); frozen_violations[edge_id]=raw_violation.copy()
                        step["topology_refreshes"]+=1
                    active=frozen_masks[edge_id].copy()
                elif topology_mode=="oracle_final":
                    if oracle_masks is None or edge_id not in oracle_masks:
                        raise RuntimeError(f"missing oracle-final topology for {edge_id}")
                    active=oracle_masks[edge_id].copy()
                    if edge_id not in frozen_masks:
                        frozen_masks[edge_id]=active.copy(); step["topology_refreshes"]+=1
                elif topology_mode=="periodic":
                    last=last_topology_refresh.get(edge_id)
                    if last is None or reverse_step-last>=topology_refresh_interval:
                        frozen_masks[edge_id]=raw_active.copy(); last_topology_refresh[edge_id]=reverse_step
                        step["topology_refreshes"]+=1
                    active=frozen_masks[edge_id].copy()
                elif topology_mode=="adaptive_hysteresis":
                    raw_prior=topology_raw_previous.get(edge_id)
                    if raw_prior is None:
                        instantaneous_volatility=0.0
                    else:
                        instantaneous_volatility=float(np.mean(raw_active!=raw_prior))
                    old_volatility=topology_volatility.get(edge_id,instantaneous_volatility)
                    smoothed_volatility=(volatility_ema*old_volatility+
                                         (1.0-volatility_ema)*instantaneous_volatility)
                    topology_volatility[edge_id]=smoothed_volatility
                    topology_raw_previous[edge_id]=raw_active.copy()
                    persistence=3 if smoothed_volatility>=volatility_threshold else 1
                    adaptive_persistence_counts[persistence]+=1
                    if edge_id not in frozen_masks:
                        frozen_masks[edge_id]=raw_active.copy()
                        topology_mismatch_streaks[edge_id]=np.zeros(samples,dtype=np.int16)
                        step["topology_refreshes"]+=1
                    else:
                        accepted=frozen_masks[edge_id]
                        prior_streak=topology_mismatch_streaks[edge_id]
                        mismatch=raw_active!=accepted
                        streak=np.where(mismatch,prior_streak+1,0).astype(np.int16)
                        commit=streak>=persistence
                        if commit.any():
                            accepted[commit]=raw_active[commit]
                            streak[commit]=0
                            step["topology_refreshes"]+=1
                        topology_mismatch_streaks[edge_id]=streak
                    active=frozen_masks[edge_id].copy()
                elif topology_mode=="row_ema_hysteresis":
                    # Track instability independently for every row-edge pair.
                    # A volatile row must repeat the same topology change before
                    # it is committed; a stable row can switch immediately.
                    raw_prior=topology_raw_previous.get(edge_id)
                    previous_ema=topology_row_volatility.get(
                        edge_id,np.zeros(samples,dtype=np.float32))
                    flip=(raw_active!=raw_prior).astype(np.float32) if raw_prior is not None else np.zeros(samples,dtype=np.float32)
                    row_ema=(volatility_ema*previous_ema+(1.0-volatility_ema)*flip).astype(np.float32)
                    topology_row_volatility[edge_id]=row_ema
                    topology_volatility[edge_id]=float(row_ema.mean())
                    topology_raw_previous[edge_id]=raw_active.copy()
                    persistence=np.where(row_ema>=volatility_threshold,3,1).astype(np.int16)
                    adaptive_persistence_counts[1]+=int((persistence==1).sum())
                    adaptive_persistence_counts[3]+=int((persistence==3).sum())
                    if edge_id not in frozen_masks:
                        frozen_masks[edge_id]=raw_active.copy()
                        topology_mismatch_streaks[edge_id]=np.zeros(samples,dtype=np.int16)
                        step["topology_refreshes"]+=1
                    else:
                        accepted=frozen_masks[edge_id]
                        prior_streak=topology_mismatch_streaks[edge_id]
                        mismatch=raw_active!=accepted
                        streak=np.where(mismatch,prior_streak+1,0).astype(np.int16)
                        commit=streak>=persistence
                        if commit.any():
                            accepted[commit]=raw_active[commit]
                            streak[commit]=0
                            step["topology_refreshes"]+=1
                        topology_mismatch_streaks[edge_id]=streak
                    active=frozen_masks[edge_id].copy()
                elif topology_mode=="hysteresis":
                    if edge_id not in frozen_masks:
                        frozen_masks[edge_id]=raw_active.copy()
                        topology_mismatch_streaks[edge_id]=np.zeros(samples,dtype=np.int16)
                        step["topology_refreshes"]+=1
                    else:
                        accepted=frozen_masks[edge_id]
                        prior_streak=topology_mismatch_streaks[edge_id]
                        mismatch=raw_active!=accepted
                        streak=np.where(mismatch,prior_streak+1,0).astype(np.int16)
                        commit=streak>=topology_persistence
                        if commit.any():
                            accepted[commit]=raw_active[commit]
                            streak[commit]=0
                            step["topology_refreshes"]+=1
                        topology_mismatch_streaks[edge_id]=streak
                    active=frozen_masks[edge_id].copy()
                elif topology_mode=="shuffled":
                    # Preserve edge support while breaking row-edge semantics.
                    active=np.roll(raw_active,1); step["topology_refreshes"]+=1
                else:
                    active=raw_active; step["topology_refreshes"]+=1
                violation=(frozen_violations[edge_id].copy() if topology_mode=="frozen_state" else active & ~satisfied)
                # Row-edge primal-dual controller. The multiplier grows only
                # while an applicable edge exceeds its calibrated residual and
                # decays after satisfaction/deactivation. It therefore changes
                # both across rows and along the reverse trajectory.
                dual=dual_state[edge_id]
                dual*=dual_decay
                dual+=dual_lr*(violation.astype(np.float32)-calibration_rate*active.astype(np.float32))
                dual[~active]*=dual_decay
                np.clip(dual,0.0,dual_max,out=dual)
                step["dual_mass"]+=float(dual.sum())
                step["dual_nonzero"]+=int((dual>0).sum())
                step["dual_max"]=max(step["dual_max"],float(dual.max(initial=0.0)))
                prior=previous_masks.get(edge_id,np.zeros(samples,dtype=bool))
                topology_prior=topology_previous.get(edge_id)
                if topology_prior is not None:
                    changed=active!=topology_prior; topology_changed_rows|=changed
                    additions=int((active & ~topology_prior).sum())
                    removals=int((~active & topology_prior).sum())
                    topology_changes_by_edge[edge_id]+=int(changed.sum())
                    step["topology_additions"]+=additions; step["topology_removals"]+=removals
                    step["topology_flips"]+=additions+removals
                topology_previous[edge_id]=active.copy(); step["active_row_edges"]+=int(active.sum())
                current_active_by_edge[edge_id]=active.copy()
                violation_prior=violation_previous.get(edge_id)
                if violation_prior is not None:
                    state_changed=violation!=violation_prior; constraint_state_changed_rows|=state_changed
                    constraint_state_changes_by_edge[edge_id]+=int(state_changed.sum())
                violation_previous[edge_id]=violation.copy()
                prior_streak=violation_streaks.get(edge_id,np.zeros(samples,dtype=np.int16))
                violation_streak=np.where(violation,prior_streak+1,0).astype(np.int16)
                violation_streaks[edge_id]=violation_streak
                stable=np.zeros(samples,dtype=bool) if previous_numeric is None else np.ones(samples,dtype=bool)
                if previous_numeric is not None:
                    involved={a["left_column"] for a in edge.get("applicability",[])+edge.get("requirements",[])}
                    involved|={a["right_column"] for a in edge.get("requirements",[]) if a.get("right_kind")=="column"}
                    for column in involved:
                        if column in layer.num_pos:
                            pos=layer.num_pos[column]; stable &= np.abs(current_numeric[:,pos]-previous_numeric[:,pos])<=drift
                gate=violation & active
                # Keep graph-state estimation separate from the action trigger.
                # `stable` rejects one-step preview errors; `transition` acts
                # only when an edge enters the violated state after entry.
                if state_gate=="stable":
                    gate &= violation_streak>=2
                elif state_gate=="stable3":
                    gate &= violation_streak>=3
                elif state_gate=="stable4":
                    gate &= violation_streak>=4
                elif state_gate in {"stable5","stable8","stable10","stable12"}:
                    required_streak=int(state_gate.removeprefix("stable"))
                    step["gate_required_streak"]=required_streak
                    gate &= violation_streak>=required_streak
                elif state_gate in {"annealed5","annealed8","annealed12"}:
                    # Early clean previews need repeated evidence, whereas late
                    # previews are already highly readable.  Anneal the required
                    # consecutive violation count from K at the beginning of the
                    # active window to one at its end.  This raises late coverage
                    # without granting noisy middle states immediate authority.
                    maximum=int(state_gate.removeprefix("annealed"))
                    width=max(active_max_t-active_min_t,1e-8)
                    remaining=float(np.clip((float(t)-active_min_t)/width,0.0,1.0))
                    required_streak=max(1,int(math.ceil(1.0+(maximum-1)*remaining)))
                    step["gate_required_streak"]=required_streak
                    gate &= violation_streak>=required_streak
                elif state_gate=="transition":
                    gate &= (~violation_prior if violation_prior is not None else np.zeros(samples,dtype=bool))
                elif state_gate=="dual":
                    gate &= dual>=dual_threshold
                if promoted:
                    # A guardrail becomes actionable only after a persistent
                    # row-level violation, and calibrated_gate below repairs
                    # only mass beyond the legitimate train exception rate.
                    gate &= violation_streak>=promoted_min_streak
                if gate_mode=="persistent": gate &= prior & stable
                if int(active.sum())<min_support: gate[:]=False
                trajectory_precision=1.0; trajectory_recall=1.0
                if trajectory_profiles and edge_id in trajectory_profiles:
                    profile=trajectory_profiles[edge_id]
                    nearest=min(profile,key=lambda item:abs(item[0]-float(t)))
                    trajectory_precision=nearest[1]; trajectory_recall=nearest[2]
                row_posterior=None
                if (gate_ranking in {"learned_posterior","posterior_cost"} and trajectory_certifiers and
                        edge_id in trajectory_certifiers):
                    # Predict whether a violation visible now will persist to
                    # the final sample.  The model was fit on independently
                    # generated trajectories and sees the complete predicted-
                    # clean mixed-type state, timestep, and violation streak.
                    encoded_state=np.concatenate([
                        current_numeric,
                        codes.detach().cpu().numpy().astype(np.float32),
                        np.full((samples,1),float(t),dtype=np.float32),
                        np.log1p(violation_streak).astype(np.float32)[:,None],
                    ],axis=1)
                    row_posterior=trajectory_certifiers[edge_id].predict_proba(encoded_state)[:,1]
                elif streak_profiles and edge_id in streak_profiles:
                    profile=streak_profiles[edge_id]
                    nearest_t=min({item[0] for item in profile},key=lambda value:abs(value-float(t)))
                    lookup={item[1]:item[2] for item in profile if item[0]==nearest_t}
                    row_posterior=np.asarray([
                        lookup.get(min(8,int(value)),-math.inf) for value in violation_streak
                    ],dtype=float)
                gate=calibrated_gate(frame,edge,gate,layer,trajectory_precision,trajectory_recall,
                                     gate_ranking,row_posterior)
                if max_actions_per_row_edge>0:
                    gate &= action_counts[edge_id]<max_actions_per_row_edge
                if cooldown_steps>0:
                    gate &= (reverse_step-last_action[edge_id])>cooldown_steps
                if max_total_actions>0:
                    remaining=max_total_actions-stats["action_events"]
                    if remaining<=0:
                        gate[:]=False
                    elif int(gate.sum())>remaining:
                        indices=np.flatnonzero(gate)
                        severity=layer.severity(frame,edge)[indices]
                        chosen=indices[np.argsort(-severity)[:remaining]]
                        gate[:]=False; gate[chosen]=True
                previous_masks[edge_id]=active
                if layer.action_planner=="component_beam":
                    pending_component_gates.append((edge,promoted,gate.copy()))
                    continue
                edge_actions=0
                for row_index in np.flatnonzero(gate):
                    original=corrected.iloc[[row_index]].copy()
                    label,candidate,columns,_=layer.choose(original,edge,action_mode)
                    if not columns: continue
                    if claimed[row_index].intersection(columns): stats["conflicts"]+=1; continue
                    claimed[row_index].update(columns)
                    if not rollback_mode:
                        corrected.loc[corrected.index[row_index],columns]=candidate.iloc[0][columns].to_numpy()
                    for column in columns:
                        if column in layer.num_pos:
                            numeric_pos=layer.num_pos[column]
                            changed_numeric[row_index,numeric_pos]=True
                            if numeric_guidance_mode=="primal_dual":
                                class_scale=min(max_guidance_scale,
                                                numeric_guidance_scale*(1.0+float(dual[row_index])))
                            elif numeric_guidance_mode=="feedback_adaptive":
                                # The clean preview becomes more reliable late in the
                                # active window.  Recurrent row-edge violations receive
                                # progressively stronger guidance, while first-time and
                                # early corrections stay conservative.
                                width=max(active_max_t-active_min_t,1e-8)
                                progress=float(np.clip((active_max_t-float(t))/width,0.0,1.0))
                                repeats=int(action_counts[edge_id][row_index])
                                time_factor=1.0+(feedback_late_multiplier-1.0)*progress
                                class_scale=min(max_guidance_scale,
                                                numeric_guidance_scale*time_factor*
                                                (feedback_repeat_growth**repeats))
                            else:
                                class_scale={"hard":hard_guidance_scale,"near_hard":near_guidance_scale,
                                             "soft":soft_guidance_scale}.get(edge.get("classification"),near_guidance_scale)
                            cell_guidance_scale[row_index,numeric_pos]=float(class_scale)
                        elif column in layer.cat_pos: changed_cat[row_index,layer.cat_pos[column]]=True
                        if action_hold_steps>0 and not rollback_mode:
                            action_holds[(edge_id,int(row_index),column)]={
                                "value":candidate.iloc[0][column],
                                "until":int(reverse_step+action_hold_steps),
                                "guidance_scale":float(class_scale) if column in layer.num_pos else 1.0,
                            }
                    choice_label=f"rollback_{label}" if rollback_mode else label
                    choices[choice_label]+=1; ever[row_index]=True; edge_actions+=1
                    lookahead_targets[row_index].append(edge_id)
                    lookahead_labels[row_index].append(choice_label)
                    lookahead_event_counts[row_index]+=1
                    action_counts[edge_id][row_index]+=1; last_action[edge_id][row_index]=reverse_step
                if edge_actions:
                    stats["actions_by_edge"][edge_id]=stats["actions_by_edge"].get(edge_id,0)+edge_actions
                    stats["action_events"]+=edge_actions; step["actions"]+=edge_actions; step["edges"]+=1
                    if promoted: stats["promoted_action_events"]+=edge_actions
            if layer.action_planner=="component_beam" and pending_component_gates:
                gated_rows=sorted(set(np.concatenate([
                    np.flatnonzero(gate) for _,_,gate in pending_component_gates if gate.any()
                ]).astype(int).tolist())) if any(gate.any() for _,_,gate in pending_component_gates) else []
                if max_total_actions>0:
                    # Edge gates are assembled before joint component planning, so
                    # per-edge truncation alone can exceed a global component budget.
                    # Rank eligible rows by their strongest current violation and
                    # enforce the cap again after the joint plan is known.
                    def row_priority(row_index):
                        return max((float(layer.severity(frame,edge)[row_index])
                                    for edge,_,gate in pending_component_gates if bool(gate[row_index])),
                                   default=0.0)
                    gated_rows=sorted(gated_rows,key=row_priority,reverse=True)
                for row_index in gated_rows:
                    if max_total_actions>0 and stats["action_events"]>=max_total_actions:
                        break
                    targets=[(edge,promoted) for edge,promoted,gate in pending_component_gates if bool(gate[row_index])]
                    if layer.component_min_edges>1:
                        components=layer.edge_components([edge for edge,_ in targets])
                        eligible_ids={edge["candidate_id"] for component in components
                                      if len(component)>=layer.component_min_edges for edge in component}
                        deferred=len(targets)-sum(edge["candidate_id"] in eligible_ids for edge,_ in targets)
                        if deferred:
                            stats["structural_deferred_edge_events"]+=int(deferred)
                            stats["structural_deferred_rows"]+=1
                        targets=[item for item in targets if item[0]["candidate_id"] in eligible_ids]
                        if not targets: continue
                    target_edges=[edge for edge,_ in targets]
                    original=corrected.iloc[[row_index]].copy()
                    candidate,label_map,columns,_,component_count=layer.choose_component(original,target_edges)
                    if not columns or component_count==0: continue
                    if max_total_actions>0 and stats["action_events"]+component_count>max_total_actions:
                        continue
                    if not rollback_mode:
                        corrected.loc[corrected.index[row_index],columns]=candidate.iloc[0][columns].to_numpy()
                    resolved=[]
                    for edge,promoted in targets:
                        edge_id=edge["candidate_id"]
                        before_bad=bool(edge_masks(original,edge)[2][0])
                        after_bad=bool(edge_masks(candidate,edge)[2][0])
                        if before_bad and not after_bad:
                            resolved.append((edge,promoted))
                            action_counts[edge_id][row_index]+=1
                            last_action[edge_id][row_index]=reverse_step
                            stats["actions_by_edge"][edge_id]=stats["actions_by_edge"].get(edge_id,0)+1
                            if promoted: stats["promoted_action_events"]+=1
                            choices[label_map.get(edge_id,"component_joint")]+=1
                    if not resolved: continue
                    for column in columns:
                        if column in layer.num_pos:
                            numeric_pos=layer.num_pos[column]; changed_numeric[row_index,numeric_pos]=True
                            if numeric_guidance_mode=="primal_dual":
                                edge_scales=[numeric_guidance_scale*(1.0+float(dual_state[edge["candidate_id"]][row_index]))
                                             for edge,_ in resolved]
                                class_scale=min(max_guidance_scale,max(edge_scales,default=numeric_guidance_scale))
                            elif numeric_guidance_mode=="feedback_adaptive":
                                width=max(active_max_t-active_min_t,1e-8)
                                progress=float(np.clip((active_max_t-float(t))/width,0.0,1.0))
                                edge_scales=[]
                                for edge,_ in resolved:
                                    repeats=max(0,int(action_counts[edge["candidate_id"]][row_index])-1)
                                    edge_scales.append(numeric_guidance_scale*
                                                       (1.0+(feedback_late_multiplier-1.0)*progress)*
                                                       (feedback_repeat_growth**repeats))
                                class_scale=min(max_guidance_scale,max(edge_scales,default=numeric_guidance_scale))
                            else:
                                class_scale=max({"hard":hard_guidance_scale,"near_hard":near_guidance_scale,
                                                 "soft":soft_guidance_scale}.get(edge.get("classification"),near_guidance_scale)
                                                for edge,_ in resolved)
                            cell_guidance_scale[row_index,numeric_pos]=float(class_scale)
                        elif column in layer.cat_pos:
                            changed_cat[row_index,layer.cat_pos[column]]=True
                        if action_hold_steps>0 and not rollback_mode:
                            hold_scale=(float(cell_guidance_scale[row_index,layer.num_pos[column]])
                                        if column in layer.num_pos else 1.0)
                            for resolved_edge,_ in resolved:
                                action_holds[(resolved_edge["candidate_id"],int(row_index),column)]={
                                    "value":candidate.iloc[0][column],
                                    "until":int(reverse_step+action_hold_steps),
                                    "guidance_scale":hold_scale,
                                }
                    ever[row_index]=True
                    lookahead_targets[row_index].extend(edge["candidate_id"] for edge,_ in resolved)
                    lookahead_labels[row_index].extend(label_map.get(edge["candidate_id"],"component_joint")
                                                       for edge,_ in resolved)
                    lookahead_event_counts[row_index]+=component_count
                    stats["component_action_events"]+=component_count
                    stats["resolved_edge_events"]+=len(resolved)
                    stats["changed_cells"]+=len(columns)
                    stats["action_events"]+=component_count
                    step["actions"]+=component_count
                    step["edges"]+=len(resolved)
            previous_numeric=current_numeric
        else:
            for edge in edges:
                previous_masks[edge["candidate_id"]]=edge_masks(frame,edge)[0]
            previous_numeric=denoised.detach().cpu().numpy()
            action_holds.clear()

        # Re-apply a chosen target for a short horizon while its edge remains
        # applicable.  This prevents the base denoiser from immediately
        # washing out a one-shot correction, but releases the target as soon
        # as the row-conditioned graph deactivates that edge.
        if action_hold_steps>0 and action_holds:
            expired=[]
            for key,hold in action_holds.items():
                edge_id,row_index,column=key
                active=current_active_by_edge.get(edge_id)
                if reverse_step>hold["until"] or active is None or not bool(active[row_index]):
                    expired.append(key); continue
                if column in layer.num_pos:
                    pos=layer.num_pos[column]
                    if changed_numeric[row_index,pos]:
                        continue
                    corrected.loc[corrected.index[row_index],column]=hold["value"]
                    changed_numeric[row_index,pos]=True
                    cell_guidance_scale[row_index,pos]=float(hold.get("guidance_scale",1.0))
                elif column in layer.cat_pos:
                    pos=layer.cat_pos[column]
                    if changed_cat[row_index,pos]:
                        continue
                    corrected.loc[corrected.index[row_index],column]=hold["value"]
                    changed_cat[row_index,pos]=True
                else:
                    continue
                step["hold_cell_applications"]+=1
            for key in expired:
                action_holds.pop(key,None)

        encoded_corrections=torch.as_tensor(layer.encode_numeric(corrected),device=device,dtype=numeric.dtype)
        correction_mask=torch.as_tensor(changed_numeric,device=device)
        # Decode/encode is not an exact identity under quantile transforms.
        # Preserve the model's x0 prediction everywhere except explicitly
        # edited cells, otherwise a no-action step itself changes the sampler.
        corrected_num=denoised.clone()
        if x0_injection and changed_numeric.any():
            if numeric_guidance_mode=="step_normalized":
                # In the deterministic Euler update, an x0 displacement is
                # attenuated by (1 - sigma_next / sigma).  Rescale the clean
                # target so each active step realizes an approximately fixed
                # fraction of the requested correction in state space.
                step_gain=(1.0-sigma_next/torch.clamp(sigma,min=1e-8)).clamp(min=1e-6)
                guidance_scale=(target_step_fraction/step_gain).clamp(max=max_guidance_scale)
            elif numeric_guidance_mode in {"hardness_adaptive","feedback_adaptive","primal_dual"}:
                guidance_scale=torch.as_tensor(cell_guidance_scale,device=device,dtype=numeric.dtype)
            else:
                guidance_scale=torch.full_like(sigma,float(numeric_guidance_scale))
            if guidance_scale.ndim==1:
                guidance_scale=guidance_scale.unsqueeze(0)
            guidance_delta=guidance_scale*(encoded_corrections-denoised)
            if max_guidance_delta>0:
                guidance_clipped_cells+=int((guidance_delta[correction_mask].abs()>max_guidance_delta).sum().item())
                guidance_delta=guidance_delta.clamp(min=-max_guidance_delta,max=max_guidance_delta)
            guided_target=denoised+guidance_delta
            corrected_num[correction_mask]=guided_target[correction_mask]
            expanded_scale=guidance_scale.expand_as(correction_mask)
            realized_guidance_scales.extend(expanded_scale[correction_mask].detach().cpu().tolist())
        else:
            corrected_num[correction_mask]=encoded_corrections[correction_mask]
        base_d=(numeric_hat-denoised)/sigma
        base_numeric_next=numeric_hat+(sigma_next-sigma)*base_d
        d=(numeric_hat-(corrected_num if x0_injection else denoised))/sigma
        numeric_next=numeric_hat+(sigma_next-sigma)*d
        if rollback_mode and changed_numeric.any():
            mask=torch.as_tensor(changed_numeric,device=device)
            rollback_noise=torch.randn(numeric_next.shape,device=device,dtype=numeric_next.dtype,generator=rollback_generator)
            rollback_state=denoised+sigma_next.unsqueeze(0)*rollback_noise
            numeric_next[mask]=rollback_state[mask]
        elif not x0_injection and changed_numeric.any():
            mask=torch.as_tensor(changed_numeric,device=device)
            numeric_next[mask]=(numeric_next+xstate_blend*(corrected_num-numeric_next))[mask]
        logits=diffusion._subs_parameterization(raw_logits,categorical_hat)
        alpha_t=torch.exp(-sc[index]).unsqueeze(0).repeat(samples,1)
        alpha_s=torch.exp(-scx[index]).unsqueeze(0).repeat(samples,1)
        categorical_next,_=diffusion._mdlm_update(logits,categorical_hat,alpha_t,alpha_s)
        base_categorical_next=categorical_next.clone()
        for column,pos in layer.cat_pos.items():
            rows=np.flatnonzero(changed_cat[:,pos])
            if len(rows):
                row_tensor=torch.as_tensor(rows,device=device)
                if rollback_mode:
                    categorical_next[row_tensor,pos]=int(diffusion.num_classes[pos])
                else:
                    vals=[layer.encode_category(column,corrected.iloc[r][column]) for r in rows]
                    categorical_next[row_tensor,pos]=torch.as_tensor(vals,device=device)
        attempted_rows=np.flatnonzero(lookahead_event_counts>0)
        attempted_events=int(lookahead_event_counts.sum())
        stats["attempted_action_events"]+=attempted_events
        if len(attempted_rows) and layer.counterfactual_lookahead and index>0 and not rollback_mode:
            cpu_rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state(device)

            def restore_rng():
                torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng,device)

            def rollout_preview(next_numeric, next_categorical):
                rollout_numeric=next_numeric.detach().clone()
                rollout_categorical=next_categorical.detach().clone()
                stop=max(-1,index-1-layer.lookahead_horizon)
                for rollout_index in range(index-1,stop,-1):
                    rollout_t=times[rollout_index]; rollout_sigma=sn[rollout_index]
                    rollout_numeric_hat=rollout_numeric+(rollout_sigma**2-rollout_sigma**2).sqrt()*torch.randn_like(rollout_numeric)
                    rollout_move=-torch.expm1(sc[rollout_index]-sc[rollout_index])
                    rollout_cat_hat,_=diffusion.q_xt(rollout_categorical,rollout_move)
                    rollout_xoh=diffusion.to_one_hot(rollout_cat_hat).to(rollout_numeric.dtype)
                    rollout_denoised,rollout_logits=preview.base(
                        rollout_numeric_hat.float(),rollout_xoh,rollout_t.squeeze().repeat(samples),
                        sigma=rollout_sigma.unsqueeze(0).repeat(samples,1))
                    rollout_codes=preview.predicted_codes(rollout_xoh,rollout_logits)
                    rollout_frame=decode(train_data,torch.cat([rollout_denoised,rollout_codes.float()],dim=1),info)
                    if rollout_index==stop+1 or rollout_index==0:
                        return rollout_frame
                    rollout_d=(rollout_numeric_hat-rollout_denoised)/rollout_sigma
                    rollout_numeric=rollout_numeric_hat+(snx[rollout_index]-rollout_sigma)*rollout_d
                    rollout_subs=diffusion._subs_parameterization(rollout_logits,rollout_cat_hat)
                    rollout_alpha_t=torch.exp(-sc[rollout_index]).unsqueeze(0).repeat(samples,1)
                    rollout_alpha_s=torch.exp(-scx[rollout_index]).unsqueeze(0).repeat(samples,1)
                    rollout_categorical,_=diffusion._mdlm_update(
                        rollout_subs,rollout_cat_hat,rollout_alpha_t,rollout_alpha_s)
                return rollout_frame

            restore_rng(); base_preview=rollout_preview(base_numeric_next,base_categorical_next)
            restore_rng(); guided_preview=rollout_preview(numeric_next,categorical_next)
            restore_rng()
            target_ids=sorted({edge_id for row in attempted_rows for edge_id in lookahead_targets[row]})
            unique_ids=(sorted(edge_by_id) if layer.lookahead_scope=="graph" else target_ids)
            class_weight={"hard":3.0,"near_hard":2.0,"soft":1.0}
            base_bad={edge_id:edge_masks(base_preview,edge_by_id[edge_id])[2] for edge_id in unique_ids}
            guided_bad={edge_id:edge_masks(guided_preview,edge_by_id[edge_id])[2] for edge_id in unique_ids}
            accepted=[]; rejected=[]
            for row_index in attempted_rows:
                scored_ids=(unique_ids if layer.lookahead_scope=="graph" else lookahead_targets[row_index])
                base_energy=sum(class_weight.get(edge_by_id[edge_id].get("classification"),1.0)*
                                float(base_bad[edge_id][row_index]) for edge_id in scored_ids)
                guided_energy=sum(class_weight.get(edge_by_id[edge_id].get("classification"),1.0)*
                                  float(guided_bad[edge_id][row_index]) for edge_id in scored_ids)
                if guided_energy+layer.lookahead_margin<base_energy:
                    accepted.append(int(row_index)); accepted_ever[row_index]=True
                else:
                    rejected.append(int(row_index))
                    numeric_next[row_index]=base_numeric_next[row_index]
                    categorical_next[row_index]=base_categorical_next[row_index]
                    events=int(lookahead_event_counts[row_index])
                    stats["action_events"]-=events; step["actions"]-=events
                    if layer.action_planner=="component_beam":
                        stats["component_action_events"]-=events
                        stats["resolved_edge_events"]-=len(lookahead_targets[row_index])
                    for edge_id,label in zip(lookahead_targets[row_index],lookahead_labels[row_index]):
                        action_counts[edge_id][row_index]=max(0,int(action_counts[edge_id][row_index])-1)
                        stats["actions_by_edge"][edge_id]=max(0,stats["actions_by_edge"].get(edge_id,0)-1)
                        if choices[label]>0: choices[label]-=1
            stats["lookahead_accepted_events"]+=int(lookahead_event_counts[accepted].sum()) if accepted else 0
            stats["lookahead_rejected_events"]+=int(lookahead_event_counts[rejected].sum()) if rejected else 0
            stats["lookahead_accepted_rows"]+=len(accepted); stats["lookahead_rejected_rows"]+=len(rejected)
            step["lookahead_accepted"]=len(accepted); step["lookahead_rejected"]=len(rejected)
        elif len(attempted_rows):
            accepted_ever[attempted_rows]=True
            stats["lookahead_accepted_events"]+=attempted_events
        numeric,categorical=numeric_next.detach(),categorical_next.detach(); stats["trace"].append(step)
    effective_ever=accepted_ever if layer.counterfactual_lookahead else ever
    stats["unique_rows_actuated"]=int(effective_ever.sum()); stats["action_choices"]={k:v for k,v in choices.items() if v}
    stats["acted_rows"]=np.flatnonzero(effective_ever).astype(int).tolist()
    stats["max_actions_per_row_edge"]=int(max_actions_per_row_edge)
    stats["cooldown_steps"]=int(cooldown_steps)
    stats["topology_mode"]=topology_mode
    stats["active_window"]=[float(active_min_t),float(active_max_t)]
    stats["topology_flips"]=int(sum(step["topology_flips"] for step in stats["trace"]))
    stats["topology_additions"]=int(sum(step["topology_additions"] for step in stats["trace"]))
    stats["topology_removals"]=int(sum(step["topology_removals"] for step in stats["trace"]))
    stats["topology_refreshes"]=int(sum(step["topology_refreshes"] for step in stats["trace"]))
    stats["topology_refresh_interval"]=int(topology_refresh_interval)
    stats["topology_persistence"]=int(topology_persistence)
    stats["volatility_threshold"]=float(volatility_threshold)
    stats["volatility_ema"]=float(volatility_ema)
    stats["topology_volatility_by_edge"]={key:float(value) for key,value in topology_volatility.items()}
    stats["final_dual_by_edge"]={key:{"mean":float(value.mean()),"max":float(value.max(initial=0.0)),
                                            "nonzero":int((value>0).sum())}
                                      for key,value in dual_state.items()}
    stats["dual_lr"]=float(dual_lr); stats["dual_decay"]=float(dual_decay)
    stats["dual_max"]=float(dual_max); stats["dual_threshold"]=float(dual_threshold)
    stats["adaptive_persistence_counts"]={str(key):int(value) for key,value in adaptive_persistence_counts.items()}
    stats["state_gate"]=state_gate
    stats["action_planner"]=layer.action_planner
    stats["component_beam_width"]=layer.component_beam_width
    stats["component_violation_weight"]=layer.component_violation_weight
    stats["lookahead_scope"]=layer.lookahead_scope
    stats["lookahead_horizon"]=layer.lookahead_horizon
    stats["max_total_actions"]=int(max_total_actions)
    stats["action_hold_steps"]=int(action_hold_steps)
    stats["hold_cell_applications"]=int(sum(step.get("hold_cell_applications",0) for step in stats["trace"]))
    stats["numeric_guidance_mode"]=numeric_guidance_mode
    stats["numeric_guidance_scale"]=float(numeric_guidance_scale)
    stats["target_step_fraction"]=float(target_step_fraction)
    stats["max_guidance_scale"]=float(max_guidance_scale)
    stats["hard_guidance_scale"]=float(hard_guidance_scale)
    stats["near_guidance_scale"]=float(near_guidance_scale)
    stats["soft_guidance_scale"]=float(soft_guidance_scale)
    stats["feedback_repeat_growth"]=float(feedback_repeat_growth)
    stats["feedback_late_multiplier"]=float(feedback_late_multiplier)
    stats["max_guidance_delta"]=float(max_guidance_delta)
    stats["guidance_clipped_cells"]=int(guidance_clipped_cells)
    stats["promote_score_only"]=bool(promote_score_only)
    stats["promoted_min_streak"]=int(promoted_min_streak)
    stats["promoted_max_calibration_rate"]=float(promoted_max_calibration_rate)
    stats["xstate_blend"]=float(xstate_blend)
    stats["gate_ranking"]=gate_ranking
    stats["realized_guidance_scale_mean"]=(float(np.mean(realized_guidance_scales)) if realized_guidance_scales else 0.0)
    stats["realized_guidance_scale_max"]=(float(np.max(realized_guidance_scales)) if realized_guidance_scales else 0.0)
    stats["topology_changed_rows"]=np.flatnonzero(topology_changed_rows).astype(int).tolist()
    stats["topology_changes_by_edge"]=dict(topology_changes_by_edge)
    stats["constraint_state_changed_rows"]=np.flatnonzero(constraint_state_changed_rows).astype(int).tolist()
    stats["constraint_state_changes_by_edge"]=dict(constraint_state_changes_by_edge)
    stats["runtime_seconds"]=time.perf_counter()-started
    return torch.cat([numeric,categorical.float()],dim=1).cpu(),stats


def final_action(frame: pd.DataFrame,edges,layer,mode,include_score_only=False,
                 max_calibration_rate=.05):
    # Final certification is the clean endpoint of the trajectory regardless
    # of which sampling mode happened to run immediately before it.
    layer.current_t=0.0
    result=frame.copy(); choices=Counter(); ever=np.zeros(len(result),dtype=bool); events=0
    eligible=[]
    for edge in edges:
        role=edge.get("usage_role", "actuate")
        calibration_rate=float(edge.get("calibration",{}).get("violation_rate",1.0))
        promoted=(include_score_only and role=="score_only" and calibration_rate<=max_calibration_rate)
        if role=="actuate" or promoted: eligible.append(edge)
    passes=0
    # Shared endpoints make a one-pass repair order-dependent.  Iterate a
    # bounded number of times so an edit for a later edge cannot silently leave
    # an earlier hard edge invalid.  The same certifier is used for proposed and
    # matched final-only repair.
    for pass_index in range(max(1,len(eligible)+1)):
        pass_events=0; passes=pass_index+1
        for edge in eligible:
            active,_,bad=edge_masks(result,edge); gate=calibrated_gate(result,edge,bad,layer)
            for row_index in np.flatnonzero(gate):
                original=result.iloc[[row_index]].copy()
                label,candidate,columns,_=layer.choose(
                    original,edge,mode,require_feasible=edge.get("classification")=="hard")
                if not columns: continue
                result.loc[result.index[row_index],columns]=candidate.iloc[0][columns].to_numpy()
                ever[row_index]=True; events+=1; pass_events+=1; choices[label]+=1
        hard_residual=sum(int(edge_masks(result,edge)[2].sum()) for edge in eligible
                          if edge.get("classification")=="hard")
        if hard_residual==0 or pass_events==0: break
    hard_residual=sum(int(edge_masks(result,edge)[2].sum()) for edge in eligible
                      if edge.get("classification")=="hard")
    return result,{"action_events":events,"unique_rows_actuated":int(ever.sum()),"action_choices":dict(choices),
                   "acted_rows":np.flatnonzero(ever).astype(int).tolist(),"runtime_seconds":0,
                   "certifier_passes":passes,"hard_residual_events":hard_residual}


def evaluate(frame,base,heldout,edges,info,method,stats,layer):
    numeric=[info["column_names"][i] for i in info["num_col_idx"]]
    categorical=[info["column_names"][i] for i in info["cat_col_idx"]]
    ks=np.mean([ks_2samp(pd.to_numeric(frame[c]),pd.to_numeric(heldout[c])).statistic for c in numeric])
    tv=[]
    for column in categorical:
        a=frame[column].astype(str).value_counts(normalize=True); b=heldout[column].astype(str).value_counts(normalize=True)
        keys=a.index.union(b.index); tv.append(.5*float((a.reindex(keys,fill_value=0)-b.reindex(keys,fill_value=0)).abs().sum()))
    drift=np.mean([np.mean(np.abs(pd.to_numeric(frame[c]).to_numpy()-pd.to_numeric(base[c]).to_numpy()))/
                   max(float(pd.to_numeric(heldout[c]).std()),1e-6) for c in numeric])
    affected=np.asarray(stats.get("acted_rows",[]),dtype=int)
    if len(affected):
        affected_drift=np.mean([np.mean(np.abs(pd.to_numeric(frame[c]).to_numpy()[affected]-pd.to_numeric(base[c]).to_numpy()[affected]))/
                                max(float(pd.to_numeric(heldout[c]).std()),1e-6) for c in numeric])
        before=base.iloc[affected][numeric].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(float)
        after=frame.iloc[affected][numeric].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(float)
        before_density=layer.knn.kneighbors(layer.scaler.transform(before),return_distance=True)[0][:,-1]/layer.density_scale
        after_density=layer.knn.kneighbors(layer.scaler.transform(after),return_distance=True)[0][:,-1]/layer.density_scale
        affected_density_delta=float(np.mean(after_density-before_density))
    else:
        affected_drift=0.0; affected_density_delta=0.0
    return {"method":method,**graph_metrics(frame,heldout,edges),**role_metrics(frame,heldout,edges),"numeric_ks":float(ks),"categorical_tv":float(np.mean(tv)),
            "numeric_drift_sd":float(drift),"action_events":stats.get("action_events",0),
            "unique_rows_actuated":stats.get("unique_rows_actuated",0),"conflicts":stats.get("conflicts",0),
            "affected_numeric_drift_sd":float(affected_drift),"affected_density_delta":affected_density_delta,
            "runtime_seconds":stats.get("runtime_seconds",0)}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,required=True); p.add_argument("--graph",type=Path,required=True)
    p.add_argument("--final-graph",type=Path,
                   help="Optional full graph for residual/final repair; middle guidance still uses --graph.")
    p.add_argument("--evaluation-graph",type=Path,
                   help="Optional frozen audit graph. Guidance uses --graph, while every reported graph metric uses this graph.")
    p.add_argument("--dataset",default="shoppers")
    p.add_argument("--schedule",type=Path,required=True); p.add_argument("--config",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--samples",type=int,default=4096); p.add_argument("--steps",type=int,default=50)
    p.add_argument("--seed",type=int,default=20261006); p.add_argument("--min-support",type=int,default=12)
    p.add_argument("--drift",type=float,default=.10); p.add_argument("--density-weight",type=float,default=1.0)
    p.add_argument("--edit-weight",type=float,default=.5)
    p.add_argument("--graph-weight",type=float,default=0.0)
    p.add_argument("--preserve-satisfied-actuate",action="store_true",
                   help="Use readable, currently satisfied actuator edges as action-selection guardrails. "
                        "They never initiate edits; they only penalize candidate actions that break them.")
    p.add_argument("--conditional-density-weight",type=float,default=0.0,
                   help="Weight of train-only local plausibility for edited cells given untouched row context.")
    p.add_argument("--conditional-neighbors",type=int,default=64)
    p.add_argument("--balanced-column-projection",action="store_true",
                   help="Add a standardized minimum-displacement two-endpoint projection candidate for numeric column relations.")
    p.add_argument("--edit-norm",choices=["l1","l2"],default="l1",
                   help="Standardized action-distance norm. Balanced projection is the exact minimum-displacement candidate under l2.")
    p.add_argument("--balanced-min-t",type=float,default=0.0,
                   help="Offer balanced projection only at t >= this value; a positive value routes late/final actions to stable endpoints.")
    p.add_argument("--balanced-routing",choices=["fixed","trajectory_reliability"],default="fixed",
                   help="Use a fixed switch or an edge-specific switch learned from train-only trajectory precision/recall.")
    p.add_argument("--action-margin-sd",type=float,default=0.0,
                   help="Move numeric inequality actions this many train standard deviations inside the feasible set.")
    p.add_argument("--action-planner",choices=["edge_local","component_beam"],default="edge_local",
                   help="Plan one edge at a time or jointly optimize each connected active subgraph.")
    p.add_argument("--lock-applicability-columns",action="store_true",
                   help="Disallow action candidates that satisfy an implication by editing its antecedent. "
                        "Use for user-specified constraints whose scope is immutable.")
    p.add_argument("--component-beam-width",type=int,default=4)
    p.add_argument("--component-violation-weight",type=float,default=100.0)
    p.add_argument("--component-min-edges",type=int,default=1,
                   help="Middle-actuate only connected gated components with at least this many edges.")
    p.add_argument("--component-prototype-k",type=int,default=0,
                   help="Add a train-only conditional median/mode prototype from K feasible context neighbours.")
    p.add_argument("--calibration-policy",choices=["point","wilson_upper","split_median","split_upper","split_wilson_upper"],default="point",
                   help="Target a calibration point or a train-split uncertainty summary. "
                        "Split policies use discovery/counterexample/calibration only.")
    p.add_argument("--calibration-z",type=float,default=0.0,
                   help="One-sided Wilson width used by uncertainty-aware near-hard calibration.")
    p.add_argument("--require-violation-evaluable",action="store_true",
                   help="Actuate only edges with enough final-positive trajectory support for violation calibration.")
    p.add_argument("--authority-mode",choices=["all","generator_excess","trajectory_precursor","graph_precursor"],default="all",
                   help="Grant middle-step authority to every catalog edge or only edges whose unguided "
                        "synthetic violation rate exceeds the train-calibrated rate.")
    p.add_argument("--authority-min-gap",type=float,default=0.0,
                   help="Minimum synthetic-minus-calibration violation-rate gap for generator_excess authority.")
    p.add_argument("--authority-min-surplus",type=float,default=0.0,
                   help="For trajectory_precursor, require an edge's max readable (middle violation count - final count) "
                        "above this value. graph_precursor grants authority to the full graph when its total positive "
                        "surplus exceeds this value.")
    p.add_argument("--trajectory-profile",type=Path,
                   help="Optional edge-state trajectory CSV used to estimate final violation mass at each step.")
    p.add_argument("--streak-profile",type=Path,
                   help="Optional calibration CSV for P(final violation | edge, timestep, violation streak).")
    p.add_argument("--trajectory-certifier",type=Path,
                   help="Optional pickle of per-edge middle-state final-violation certifiers.")
    p.add_argument("--xstate-blend",type=float,default=1.0,
                   help="Proximal interpolation strength from the unguided next state to a typed corrected state.")
    p.add_argument("--gate-ranking",choices=["severity","action_cost","trajectory_posterior","learned_posterior","posterior_cost","graph_rate"],default="severity",
                   help="Choose excess row violations by largest magnitude or lowest feasible action cost.")
    p.add_argument("--counterfactual-lookahead",action="store_true",
                   help="Accept a middle action only when its next-step preview improves target active-subgraph energy.")
    p.add_argument("--lookahead-margin",type=float,default=0.0)
    p.add_argument("--lookahead-scope",choices=["target","graph"],default="target")
    p.add_argument("--lookahead-horizon",type=int,default=1,
                   help="Number of matched reverse steps used to compare action and no-action branches.")
    p.add_argument("--max-actions-per-row-edge",type=int,default=0)
    p.add_argument("--max-total-actions",type=int,default=0,
                   help="Global action budget per guided sample; 0 disables the cap.")
    p.add_argument("--cooldown-steps",type=int,default=0)
    p.add_argument("--schedule-mode",choices=["edge","common","all"],default="edge")
    p.add_argument("--gate-mode",choices=["current","persistent"],default="persistent")
    p.add_argument("--common-cutoff",type=float,default=.306)
    p.add_argument("--topology-mode",choices=["dynamic","frozen","frozen_state","periodic","hysteresis","adaptive_hysteresis","row_ema_hysteresis","shuffled","oracle_final"],default="dynamic")
    p.add_argument("--topology-refresh-interval",type=int,default=1)
    p.add_argument("--topology-persistence",type=int,default=2)
    p.add_argument("--volatility-threshold",type=float,default=.0015)
    p.add_argument("--volatility-ema",type=float,default=.8)
    p.add_argument("--state-gate",choices=["current","stable","stable3","stable4","stable5","stable8","stable10","stable12",
                                                   "annealed5","annealed8","annealed12","transition","dual"],default="current")
    p.add_argument("--online-modes",nargs="+",choices=["xstate_directed_calibrated","xstate_reward_calibrated",
                                                        "x0_directed_calibrated","x0_reward_calibrated","rollback_reward_calibrated"],
                   default=["xstate_directed_calibrated","x0_directed_calibrated","x0_reward_calibrated","rollback_reward_calibrated"],
                   help="Online mechanisms to run; defaults preserve the original full suite.")
    p.add_argument("--active-min-t",type=float,default=.10)
    p.add_argument("--active-max-t",type=float,default=.75)
    p.add_argument("--numeric-guidance-mode",choices=["constant","step_normalized","hardness_adaptive","feedback_adaptive","primal_dual"],default="constant")
    p.add_argument("--numeric-guidance-scale",type=float,default=1.0)
    p.add_argument("--target-step-fraction",type=float,default=.10)
    p.add_argument("--max-guidance-scale",type=float,default=8.0)
    p.add_argument("--hard-guidance-scale",type=float,default=4.0)
    p.add_argument("--near-guidance-scale",type=float,default=1.0)
    p.add_argument("--soft-guidance-scale",type=float,default=.5)
    p.add_argument("--feedback-repeat-growth",type=float,default=1.5,
                   help="Multiplicative strength increase after a row-edge violates again.")
    p.add_argument("--feedback-late-multiplier",type=float,default=2.0,
                   help="Strength multiplier from the start to the end of the active window.")
    p.add_argument("--max-guidance-delta",type=float,default=0.0,
                   help="Trust-region cap on each transformed-numeric x0 displacement; 0 disables it.")
    p.add_argument("--promote-score-only",action="store_true",
                   help="Dynamically actuate persistent calibrated score-only edges.")
    p.add_argument("--promoted-min-streak",type=int,default=3)
    p.add_argument("--promoted-max-calibration-rate",type=float,default=.05)
    p.add_argument("--action-hold-steps",type=int,default=0,
                   help="Reapply a selected cell target for H subsequent reverse steps while its edge remains active.")
    p.add_argument("--dual-lr",type=float,default=.5)
    p.add_argument("--dual-decay",type=float,default=.9)
    p.add_argument("--dual-max",type=float,default=8.0)
    p.add_argument("--dual-threshold",type=float,default=1.0)
    p.add_argument("--max-actuation-calibration-rate",type=float,default=1.0,
                   help="Exclude discovered actuator edges above this train-calibration violation rate.")
    p.add_argument("--guided-final-repair",action="store_true",
                   help="Also repair the guided output and report residual repair work versus final-only repair.")
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    seed_all(a.seed); sys.path.insert(0,str(a.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model,UniModMLP
    info=json.loads((a.repo/"data"/a.dataset/"info.json").read_text()); train_data,test_data=preprocess_dataset(a.repo,a.dataset)
    train,heldout=decode(train_data,train_data.X,info),decode(train_data,test_data.X,info)
    edges=json.loads(a.graph.read_text()); schedule=pd.read_csv(a.schedule)
    # Empirically discovered relations must pass the train-calibration safety
    # threshold.  An explicit external hard specification has independent
    # authority: its empirical violation rate measures distribution shift, not
    # whether the requested condition is allowed to actuate.
    edges=[edge for edge in edges if edge.get("usage_role","actuate")!="actuate" or
           bool(edge.get("external_authority",False)) or
           float(edge.get("calibration",{}).get("violation_rate",1.0))<=a.max_actuation_calibration_rate]
    trajectory_profiles=None; trajectory_frame=None
    if a.trajectory_profile:
        trajectory_frame=pd.read_csv(a.trajectory_profile)
        trajectory_profiles={}
        for row in trajectory_frame.itertuples(index=False):
            precision=float(row.violation_precision); recall=float(row.violation_recall)
            if np.isfinite(precision) and np.isfinite(recall) and recall>0:
                trajectory_profiles.setdefault(str(row.candidate_id),[]).append(
                    (float(row.t),precision,recall))
    streak_profiles=None
    if a.streak_profile:
        streak_frame=pd.read_csv(a.streak_profile)
        streak_profiles={}
        for row in streak_frame.itertuples(index=False):
            posterior=float(row.final_violation_posterior)
            if np.isfinite(posterior) and int(row.support)>0:
                # Beta(1,1) shrinkage makes low-support trajectory states less
                # extreme while retaining a compact non-parametric certifier.
                positives=posterior*int(row.support)
                shrunk=(positives+1.0)/(int(row.support)+2.0)
                streak_profiles.setdefault(str(row.candidate_id),[]).append(
                    (float(row.t),int(row.streak_bin),float(shrunk)))
    trajectory_certifiers=None
    if a.trajectory_certifier:
        with a.trajectory_certifier.open("rb") as handle:
            payload=pickle.load(handle)
        trajectory_certifiers=payload["models"]
    cutoffs={str(r.candidate_id):float(r.cutoff) for r in schedule.itertuples(index=False)
             if bool(r.readable) and np.isfinite(r.cutoff)}
    violation_evaluable={str(r.candidate_id):bool(getattr(r,"violation_evaluable",True))
                          for r in schedule.itertuples(index=False)}
    edges=[x for x in edges if x["candidate_id"] in cutoffs and
           (not a.require_violation_evaluable or x.get("usage_role","actuate")!="actuate" or
            violation_evaluable.get(x["candidate_id"],True))]
    if a.authority_mode=="generator_excess":
        authority_gap={
            str(r.candidate_id):float(r.synthetic_violation_rate)-float(r.calibration_violation_rate)
            for r in schedule.itertuples(index=False)
            if np.isfinite(float(r.synthetic_violation_rate)) and
               np.isfinite(float(r.calibration_violation_rate))
        }
        edges=[x for x in edges if x.get("usage_role","actuate")!="actuate" or
               authority_gap.get(x["candidate_id"],-np.inf)>a.authority_min_gap]
    elif a.authority_mode in {"trajectory_precursor","graph_precursor"}:
        if trajectory_frame is None:
            raise ValueError("trajectory_precursor authority requires --trajectory-profile")
        precursor_surplus={}
        for edge in edges:
            edge_id=edge["candidate_id"]
            block=trajectory_frame[trajectory_frame.candidate_id.astype(str).eq(edge_id)].copy()
            cutoff=cutoffs.get(edge_id,-math.inf)
            block=block[
                block.t.le(cutoff+1e-8)
                & block.violation_precision.fillna(0).ge(.90)
                & block.violation_recall.fillna(0).ge(.80)
            ]
            precursor_surplus[edge_id]=(float((block.violation-block.final_violations).max())
                                         if len(block) else -math.inf)
        if a.authority_mode=="trajectory_precursor":
            edges=[x for x in edges if x.get("usage_role","actuate")!="actuate" or
                   precursor_surplus.get(x["candidate_id"],-math.inf)>a.authority_min_surplus]
        else:
            total_positive_surplus=sum(max(0.0,value) for value in precursor_surplus.values()
                                       if np.isfinite(value))
            if total_positive_surplus<=a.authority_min_surplus:
                edges=[x for x in edges if x.get("usage_role","actuate")!="actuate"]
    final_edges=json.loads(a.final_graph.read_text()) if a.final_graph else edges
    final_edges=[x for x in final_edges if
                 (x.get("usage_role","actuate")!="actuate" or bool(x.get("external_authority",False)) or
                  float(x.get("calibration",{}).get("violation_rate",1.0))<=a.max_actuation_calibration_rate) and
                 x["candidate_id"] in cutoffs and
                 (not a.require_violation_evaluable or x.get("usage_role","actuate")!="actuate" or
                  violation_evaluable.get(x["candidate_id"],True))]
    evaluation_edges=json.loads(a.evaluation_graph.read_text()) if a.evaluation_graph else edges
    with a.config.open("rb") as h: config=pickle.load(h)
    params=copy.deepcopy(config["unimodmlp_params"]); params["d_numerical"]=train_data.d_numerical; params["categories"]=(train_data.categories+1).tolist()
    model=Model(UniModMLP(**params),**config["diffusion_params"]["edm_params"])
    state=torch.load(a.checkpoint,map_location="cpu",weights_only=False); model.load_state_dict(state["denoise_fn"],strict=True)
    preview=PreviewModel(model,train_data.categories)
    dp=copy.deepcopy(config["diffusion_params"]); dp.update(num_timesteps=a.steps,scheduler="power_mean_per_column",cat_scheduler="log_linear_per_column")
    dp["sampler_params"]=copy.deepcopy(dp["sampler_params"]); dp["sampler_params"].update(stochastic_sampler=False,second_order_correction=False)
    diffusion=UnifiedCtimeDiffusion(train_data.categories,train_data.d_numerical,preview,None,**dp,device=torch.device("cuda")).cuda()
    diffusion.num_schedule.load_state_dict(state["num_schedule"]); diffusion.cat_schedule.load_state_dict(state["cat_schedule"]); diffusion.eval()
    layer=RowActionLayer(edges,train_data,train,info,a.density_weight,a.edit_weight,a.graph_weight,
                         a.action_margin_sd,a.action_planner,a.component_beam_width,
                         a.component_violation_weight,a.counterfactual_lookahead,
                         a.lookahead_margin,a.lookahead_scope,a.lookahead_horizon,
                         a.component_min_edges,a.component_prototype_k,
                         a.calibration_policy,a.calibration_z,
                         a.lock_applicability_columns,
                         a.preserve_satisfied_actuate,
                         a.conditional_density_weight,a.conditional_neighbors,
                         a.balanced_column_projection,a.edit_norm,a.balanced_min_t,
                         a.balanced_routing); common=a.seed+100000
    if a.balanced_routing=="trajectory_reliability":
        if not trajectory_profiles:
            raise ValueError("trajectory_reliability routing requires --trajectory-profile")
        for edge_id,profile in trajectory_profiles.items():
            reliable=[t for t,precision,recall in profile if precision>=.90 and recall>=.90]
            if reliable:
                layer.balanced_switch_by_edge[edge_id]=float(max(reliable))
    seed_all(common); base=decode(train_data,diffusion.sample(a.samples),info)
    sample_dir=a.output/"samples"; sample_dir.mkdir()
    base.to_csv(sample_dir/"unconditional.csv",index=False)
    rows=[evaluate(base,base,heldout,evaluation_edges,info,"unconditional",{},layer)]; edge_rows=edge_metric_rows(base,heldout,evaluation_edges,"unconditional")
    seed_all(common); parity_encoded,parity_stats=sample_guided(
        diffusion,preview,train_data,info,[],layer,{},a.samples,"x0_directed_calibrated",a.min_support,a.drift,
        a.max_actions_per_row_edge,a.cooldown_steps,a.schedule_mode,a.gate_mode,a.common_cutoff,
        a.topology_mode,a.active_min_t,a.active_max_t,a.topology_refresh_interval,a.state_gate,None,a.max_total_actions,
        a.topology_persistence,a.volatility_threshold,a.volatility_ema,
        a.numeric_guidance_mode,a.numeric_guidance_scale,a.target_step_fraction,a.max_guidance_scale,a.action_hold_steps,
        a.hard_guidance_scale,a.near_guidance_scale,a.soft_guidance_scale,
        a.feedback_repeat_growth,a.feedback_late_multiplier,a.max_guidance_delta,
        a.promote_score_only,a.promoted_min_streak,a.promoted_max_calibration_rate)
    parity=decode(train_data,parity_encoded,info)
    rows.append(evaluate(parity,base,heldout,evaluation_edges,info,"manual_no_action",parity_stats,layer))
    edge_rows+=edge_metric_rows(parity,heldout,evaluation_edges,"manual_no_action")
    oracle_masks={edge["candidate_id"]:edge_masks(base,edge)[0] for edge in edges}
    for mode in a.online_modes:
        seed_all(common); encoded,stats=sample_guided(diffusion,preview,train_data,info,edges,layer,cutoffs,a.samples,mode,a.min_support,a.drift,
                                                     a.max_actions_per_row_edge,a.cooldown_steps,a.schedule_mode,a.gate_mode,a.common_cutoff,
                                                     a.topology_mode,a.active_min_t,a.active_max_t,a.topology_refresh_interval,a.state_gate,oracle_masks,
                                                     a.max_total_actions,a.topology_persistence,
                                                     a.volatility_threshold,a.volatility_ema,
                                                     a.numeric_guidance_mode,a.numeric_guidance_scale,
                                                     a.target_step_fraction,a.max_guidance_scale,a.action_hold_steps,
                                                     a.hard_guidance_scale,a.near_guidance_scale,a.soft_guidance_scale,
                                                     a.feedback_repeat_growth,a.feedback_late_multiplier,a.max_guidance_delta,
                                                     a.promote_score_only,a.promoted_min_streak,a.promoted_max_calibration_rate,
                                                     trajectory_profiles,a.xstate_blend,a.gate_ranking,streak_profiles,
                                                     trajectory_certifiers,a.dual_lr,a.dual_decay,a.dual_max,a.dual_threshold)
        frame=decode(train_data,encoded,info); frame.to_csv(sample_dir/f"{mode}.csv",index=False)
        rows.append(evaluate(frame,base,heldout,evaluation_edges,info,mode,stats,layer)); edge_rows+=edge_metric_rows(frame,heldout,evaluation_edges,mode)
        (a.output/f"trace_{mode}.json").write_text(json.dumps(stats,indent=2))
        if a.guided_final_repair:
            # Keep the online+repair comparison symmetric.  When persistent
            # score-only edges are granted actuation authority online, the
            # residual repair must receive the same verified edge set; using
            # only the original actuators would advantage the final-only
            # promoted baseline by construction.
            repaired,repair_stats=final_action(
                frame,final_edges,layer,"reward",a.promote_score_only,
                a.promoted_max_calibration_rate,
            )
            suffix="final_promoted_reward" if a.promote_score_only else "final_reward"
            name=f"{mode}_then_{suffix}"
            combined_stats=dict(repair_stats)
            online_rows=set(stats.get("acted_rows",[])); repair_rows=set(repair_stats.get("acted_rows",[]))
            combined_stats.update({
                "online_action_events":int(stats.get("action_events",0)),
                "residual_repair_events":int(repair_stats.get("action_events",0)),
                "action_events":int(stats.get("action_events",0))+int(repair_stats.get("action_events",0)),
                "acted_rows":sorted(online_rows|repair_rows),
                "unique_rows_actuated":len(online_rows|repair_rows),
                "runtime_seconds":float(stats.get("runtime_seconds",0))+float(repair_stats.get("runtime_seconds",0)),
            })
            repaired.to_csv(sample_dir/f"{name}.csv",index=False)
            rows.append(evaluate(repaired,base,heldout,evaluation_edges,info,name,combined_stats,layer))
            edge_rows+=edge_metric_rows(repaired,heldout,evaluation_edges,name)
            (a.output/f"trace_{name}.json").write_text(json.dumps(combined_stats,indent=2))
    for action_mode in ["directed","reward"]:
        frame,stats=final_action(base,final_edges,layer,action_mode); name=f"final_{action_mode}_calibrated"
        frame.to_csv(sample_dir/f"{name}.csv",index=False)
        rows.append(evaluate(frame,base,heldout,evaluation_edges,info,name,stats,layer)); edge_rows+=edge_metric_rows(frame,heldout,evaluation_edges,name)
        (a.output/f"trace_{name}.json").write_text(json.dumps(stats,indent=2))
    if a.promote_score_only:
        frame,stats=final_action(base,final_edges,layer,"reward",True,a.promoted_max_calibration_rate)
        name="final_promoted_reward_calibrated"
        frame.to_csv(sample_dir/f"{name}.csv",index=False)
        rows.append(evaluate(frame,base,heldout,evaluation_edges,info,name,stats,layer))
        edge_rows+=edge_metric_rows(frame,heldout,evaluation_edges,name)
        (a.output/f"trace_{name}.json").write_text(json.dumps(stats,indent=2))
    metrics=pd.DataFrame(rows); metrics.to_csv(a.output/"method_metrics.csv",index=False); pd.DataFrame(edge_rows).to_csv(a.output/"edge_metrics.csv",index=False)
    (a.output/"active_graph.json").write_text(json.dumps(edges,indent=2))
    (a.output/"final_graph.json").write_text(json.dumps(final_edges,indent=2))
    (a.output/"evaluation_graph.json").write_text(json.dumps(evaluation_edges,indent=2))
    metadata={"dataset":a.dataset,"samples":a.samples,"steps":a.steps,"seed":a.seed,"edges":len(edges),
              "final_edges":len(final_edges),"final_graph":str(a.final_graph) if a.final_graph else "same_as_guidance",
              "evaluation_edges":len(evaluation_edges),"evaluation_graph":str(a.evaluation_graph) if a.evaluation_graph else "same_as_guidance",
              "density_weight":a.density_weight,
              "edit_weight":a.edit_weight,"graph_weight":a.graph_weight,"action_margin_sd":a.action_margin_sd,
              "edit_norm":a.edit_norm,"balanced_column_projection":a.balanced_column_projection,
              "balanced_min_t":a.balanced_min_t,
              "balanced_routing":a.balanced_routing,
              "balanced_switch_by_edge":layer.balanced_switch_by_edge,
              "preserve_satisfied_actuate":a.preserve_satisfied_actuate,
              "conditional_density_weight":a.conditional_density_weight,
              "conditional_neighbors":a.conditional_neighbors,
              "action_planner":a.action_planner,"component_beam_width":a.component_beam_width,
              "lock_applicability_columns":a.lock_applicability_columns,
              "component_violation_weight":a.component_violation_weight,
              "component_min_edges":a.component_min_edges,
              "component_prototype_k":a.component_prototype_k,
              "calibration_policy":a.calibration_policy,"calibration_z":a.calibration_z,
              "trajectory_certifier":str(a.trajectory_certifier) if a.trajectory_certifier else None,
              "trajectory_certifier_edges":sorted(trajectory_certifiers) if trajectory_certifiers else [],
              "require_violation_evaluable":a.require_violation_evaluable,
              "authority_mode":a.authority_mode,"authority_min_gap":a.authority_min_gap,
              "authority_min_surplus":a.authority_min_surplus,
              "precursor_surplus":precursor_surplus if a.authority_mode in {"trajectory_precursor","graph_precursor"} else {},
              "graph_precursor_total":(sum(max(0.0,value) for value in precursor_surplus.values()
                                             if np.isfinite(value))
                                        if a.authority_mode in {"trajectory_precursor","graph_precursor"} else None),
              "trajectory_profile":str(a.trajectory_profile) if a.trajectory_profile else None,
              "streak_profile":str(a.streak_profile) if a.streak_profile else None,
              "xstate_blend":a.xstate_blend,
              "gate_ranking":a.gate_ranking,
              "counterfactual_lookahead":a.counterfactual_lookahead,
              "lookahead_margin":a.lookahead_margin,
              "lookahead_scope":a.lookahead_scope,
              "lookahead_horizon":a.lookahead_horizon,
              "mechanism":"x0 clean-preview correction and row-wise branch-score-select",
              "reward":"typed certifier + frozen train-kNN density + normalized edit cost",
              "max_actions_per_row_edge":a.max_actions_per_row_edge,"cooldown_steps":a.cooldown_steps,
              "max_total_actions":a.max_total_actions,
              "numeric_guidance_mode":a.numeric_guidance_mode,"numeric_guidance_scale":a.numeric_guidance_scale,
              "target_step_fraction":a.target_step_fraction,"max_guidance_scale":a.max_guidance_scale,
              "action_hold_steps":a.action_hold_steps,
              "guided_final_repair":a.guided_final_repair,
              "hard_guidance_scale":a.hard_guidance_scale,"near_guidance_scale":a.near_guidance_scale,
              "soft_guidance_scale":a.soft_guidance_scale,
              "feedback_repeat_growth":a.feedback_repeat_growth,
              "feedback_late_multiplier":a.feedback_late_multiplier,
              "max_guidance_delta":a.max_guidance_delta,
              "promote_score_only":a.promote_score_only,
              "promoted_min_streak":a.promoted_min_streak,
              "promoted_max_calibration_rate":a.promoted_max_calibration_rate,
              "max_actuation_calibration_rate":a.max_actuation_calibration_rate,
              "dual_lr":a.dual_lr,"dual_decay":a.dual_decay,
              "dual_max":a.dual_max,"dual_threshold":a.dual_threshold,
              "online_modes":a.online_modes,
              "schedule_mode":a.schedule_mode,"gate_mode":a.gate_mode,"common_cutoff":a.common_cutoff,"api_calls":0}
    metadata.update({"topology_mode":a.topology_mode,"active_min_t":a.active_min_t,"active_max_t":a.active_max_t,
                     "topology_refresh_interval":a.topology_refresh_interval,"topology_persistence":a.topology_persistence,
                     "volatility_threshold":a.volatility_threshold,"volatility_ema":a.volatility_ema,
                     "state_gate":a.state_gate})
    (a.output/"run_metadata.json").write_text(json.dumps(metadata,indent=2))
    lines=["# Row-wise reverse action guidance","",f"Settings: `{metadata}`","",
           "| Method | Violation | Heldout error | Calibration error | Numeric KS | TV | Drift | Affected drift | Density delta | Actions | Rows | Runtime |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in metrics.itertuples(index=False):
        lines.append(f"| {r.method} | {r.macro_violation:.4f} | {r.heldout_violation_error:.4f} | {r.calibration_violation_error:.4f} | {r.numeric_ks:.4f} | {r.categorical_tv:.4f} | {r.numeric_drift_sd:.4f} | {r.affected_numeric_drift_sd:.4f} | {r.affected_density_delta:.4f} | {r.action_events} | {r.unique_rows_actuated} | {r.runtime_seconds:.2f} |")
    (a.output/"RESULTS.md").write_text("\n".join(lines)+"\n"); print("\n".join(lines))


if __name__=="__main__": main()
