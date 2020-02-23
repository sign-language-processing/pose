import {Component, Prop, h, State} from '@stencil/core';

import {
  Pose,
  PoseModel,
  PosePointModel,
  RGBColor,
  PoseHeaderComponentModel
} from "pose-utils";


@Component({
  tag: 'pose-viewer',
  styleUrl: 'pose-viewer.css',
  shadow: true
})
export class PoseViewer {
  /**
   * Pose Img Source
   */
  @Prop() src: string;

  /**
   * Allow editing the img
   */
  @Prop() edit: boolean = false;

  @State() pose: PoseModel;


  constructor() {
  }

  async componentWillLoad() {
    this.pose = await Pose.fromRemote(this.src);
  }


  isJointValid(joint: PosePointModel) {
    return joint.C > 0;
  }

  animate(component: string, i: number, attributes: any) {
    const jointsFrames: PosePointModel[][] = this.pose.body.frames.map(f => f.people[0][component]);
    if (jointsFrames.length === 1) {
      return;
    }
    const msfp = 1000 / this.pose.body.fps;

    return jointsFrames.map((frame, j) => {
      const prevIndex = j - 1 < 0 ? jointsFrames.length - 1 : j - 1;
      const nextIndex = (j + 1) % jointsFrames.length;
      const nextFrame = jointsFrames[nextIndex];
      const begin = (d: string) => (j === 0 ? "0s;" : "") + `frame_${d}_${prevIndex}.end`;

      return Object.entries(attributes).map(([attr, k]: [string, string]) => {
        return <animate id={`frame_${attr}_${j}`} attributeName={attr} from={frame[i][k]} to={nextFrame[i][k]}
                        dur={msfp + "ms"}
                        begin={begin(attr)} repeatCount="1" fill="freeze"/>
      });
    });
  }

  animateJoint(component: string, i: number) {
    return this.animate(component, i, {
      cx: "X",
      cy: "Y"
    });
  }

  renderJoints(component: string, joints: PosePointModel[], colors: RGBColor[]) {
    return joints
      .filter(this.isJointValid.bind(this))
      .map((joint, i) => {
        const {R, G, B} = colors[i % colors.length];
        return (<circle
          cx={joint.X}
          cy={joint.Y}
          r={4}
          class="joint draggable"
          style={{
            fill: `rgb(${R}, ${G}, ${B})`,
            opacity: String(joint.C)
          }}>
          {this.animateJoint(component, i)}
        </circle>);
      });
  }

  animateLimb(component: string, i: number, j: number) {
    return [
      this.animate(component, i, {
        x1: "X",
        y1: "Y"
      }),
      this.animate(component, j, {
        x2: "X",
        y2: "Y"
      })
    ]
  }

  renderLimbs(component: PoseHeaderComponentModel, joints: PosePointModel[]) {
    return component.limbs.map(({from, to}) => {
      const a = joints[from];
      const b = joints[to];
      if (!this.isJointValid(a) || !this.isJointValid(b)) {
        return "";
      }

      const c1 = component.colors[from % component.colors.length];
      const c2 = component.colors[to % component.colors.length];
      const {R, G, B} = {
        R: (c1.R + c2.R) / 2,
        G: (c1.G + c2.G) / 2,
        B: (c1.B + c2.B) / 2,
      };

      return (<line
        x1={joints[from].X}
        y1={joints[from].Y}
        x2={joints[to].X}
        y2={joints[to].Y}
        style={{
          stroke: `rgb(${R}, ${G}, ${B})`,
          opacity: String((joints[from].C + joints[to].C) / 2)
        }}>
        {this.animateLimb(component.name, from, to)}
      </line>);
    });
  }


  render() {
    if (!this.pose) {
      return "";
    }
    const startingFrame = this.pose.body.frames[0];

    return (
      <svg xmlns="http://www.w3.org/2000/svg" width={this.pose.header.width} height={this.pose.header.height}>
        <g>
          {startingFrame.people.map(person => this.pose.header.components.map(component => {
            const joints = person[component.name];
            return [
              this.renderLimbs(component, joints),
              this.renderJoints(component.name, joints, component.colors),
            ]
          }))}
        </g>
      </svg>
    )
  }
}


