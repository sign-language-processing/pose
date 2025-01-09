import {PoseViewer} from "../pose-viewer";
import {PoseBodyFrameModel, PoseLimb, PosePointModel, RGBColor} from "pose-format";

export abstract class PoseRenderer {

  constructor(protected viewer: PoseViewer) {
  }

  x(v: number) {
    const n = v * (this.viewer.elWidth - 2 * this.viewer.elPadding.width);
    return n / this.viewer.pose.header.width + this.viewer.elPadding.width;
  }

  y(v: number) {
    const n = v * (this.viewer.elHeight - 2 * this.viewer.elPadding.height);
    return n / this.viewer.pose.header.height + this.viewer.elPadding.height;
  }

  isJointValid(joint: PosePointModel) {
    return joint.C > 0;
  }

  abstract renderJoint(i: number, joint: PosePointModel, color: RGBColor);

  renderJoints(joints: PosePointModel[], colors: RGBColor[]) {
    return joints
      .filter(this.isJointValid.bind(this))
      .map((joint, i) => {
        return this.renderJoint(i, joint, colors[i % colors.length]);
      });
  }

  abstract renderLimb(from: PosePointModel, to: PosePointModel, color: RGBColor);

  renderLimbs(limbs: PoseLimb[], joints: PosePointModel[], colors: RGBColor[]) {
    /**
     This implementation is a bit different from the python one.
     In python, we sort all limbs of all people and all components by depth and then render them.
     Here, we only sort the limbs of the current component by depth.
     */

    const lines = limbs.map(({from, to}) => {
      const a = joints[from];
      const b = joints[to];
      if (!this.isJointValid(a) || !this.isJointValid(b)) {
        return null;
      }

      const c1 = colors[from % colors.length];
      const c2 = colors[to % colors.length];
      const color = {
        R: (c1.R + c2.R) / 2,
        G: (c1.G + c2.G) / 2,
        B: (c1.B + c2.B) / 2,
      };

      return {from: a, to: b, color, z: (a.Z + b.Z) / 2};
    });

    return lines
      .filter(Boolean) // Remove invalid lines
      .sort((a, b) => b.z - a.z) // Sort lines by depth
      .map(({from, to, color}) => this.renderLimb(from, to, color));
  }

  renderFrame(frame: PoseBodyFrameModel) {
    return frame.people.map(person => this.viewer.pose.header.components.map(component => {
      const joints = person[component.name];
      return [
        this.renderJoints(joints, component.colors),
        this.renderLimbs(component.limbs, joints, component.colors),
      ]
    }))
  }

  abstract render(frame: PoseBodyFrameModel);
}
