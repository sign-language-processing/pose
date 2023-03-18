import {PoseBodyFrameModel, PosePointModel, RGBColor} from "pose-format";
import {PoseRenderer} from "./pose-renderer";
import {h} from '@stencil/core';

export class SVGPoseRenderer extends PoseRenderer {

  renderJoint(i: number, joint: PosePointModel, color: RGBColor) {
    const {R, G, B} = color;

    return (<circle
      cx={joint.X}
      cy={joint.Y}
      r={4}
      class="joint draggable"
      style={({
        fill: `rgb(${R}, ${G}, ${B})`,
        opacity: String(joint.C)
      })}
      data-id={i}>
    </circle>);
  }

  renderLimb(from: PosePointModel, to: PosePointModel, color: RGBColor) {
    const {R, G, B} = color;

    return (<line
      x1={from.X}
      y1={from.Y}
      x2={to.X}
      y2={to.Y}
      style={{
        stroke: `rgb(${R}, ${G}, ${B})`,
        opacity: String((from.C + to.C) / 2)
      }}>
    </line>);
  }

  render(frame: PoseBodyFrameModel) {
    const viewBox = `0 0 ${this.viewer.pose.header.width} ${this.viewer.pose.header.height}`;
    return (
      <svg xmlns="http://www.w3.org/2000/svg"
           viewBox={viewBox}
           width={this.viewer.elWidth}
           height={this.viewer.elHeight}>
        <g>
          {this.renderFrame(frame)}
        </g>
      </svg>
    )
  }
}
