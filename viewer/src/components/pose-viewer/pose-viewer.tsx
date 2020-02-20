import {Component, Prop, h, State} from '@stencil/core';
import {parsePose, PoseBodyFrameModel, PoseLimb, PoseModel, PosePointModel, RGBColor} from "../../utils/binary-parser";
import {Buffer} from "buffer";


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

  pose: PoseModel;

  nextFrameId = 0;
  @State() frame: PoseBodyFrameModel;
  private loopInterval: NodeJS.Timeout;

  constructor() {
  }

  async componentWillLoad() {
    const res = await fetch(this.src);
    const buffer = Buffer.from(await res.arrayBuffer());
    this.pose = parsePose(buffer);

    let frame = 0;

    this.frame = this.pose.body.frames[frame - 1];

    if (this.pose.body.frames.length > 1) {
      this.clearInterval();
      this.loopInterval = setInterval(this.frameLoop.bind(this), 50)
    } else {
      this.frameLoop();
    }
  }

  clearInterval() {
    console.log(this.loopInterval);
    if (this.loopInterval) {
      clearInterval(this.loopInterval);
    }
  }

  componentDidUnload() {
    this.clearInterval();
  }

  frameLoop() {
    this.frame = this.pose.body.frames[this.nextFrameId];
    this.nextFrameId = ((this.nextFrameId + 1) % this.pose.body.frames.length);
  }

  isJointValid(joint: PosePointModel) {
    return joint.X !== 0 && joint.Y !== 0;
  }

  renderJoints(joints: PosePointModel[], colors: RGBColor[]) {
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
          }}
          data-id={i}>
        </circle>);
      });
  }

  renderLimbs(limbs: PoseLimb[], joints: PosePointModel[], colors: RGBColor[]) {
    return limbs.map(({from, to}) => {
      const a = joints[from];
      const b = joints[to];
      if (!this.isJointValid(a) || !this.isJointValid(b)) {
        return "";
      }

      const c1 = colors[from % colors.length];
      const c2 = colors[to % colors.length];
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
      </line>);
    });
  }


  render() {
    if (!this.pose) {
      return "";
    }

    return (
      <svg xmlns="http://www.w3.org/2000/svg" width={this.pose.header.width} height={this.pose.header.height}>
        <g>
          {this.frame.people.map(person => this.pose.header.components.map(component => {
            const joints = person[component.name];
            return [
              this.renderLimbs(component.limbs, joints, component.colors),
              this.renderJoints(joints, component.colors),
            ]
          }))}
        </g>
      </svg>
    )
  }
}


