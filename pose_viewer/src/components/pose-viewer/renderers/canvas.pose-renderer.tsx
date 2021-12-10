import {PoseBodyFrameModel, PosePointModel, RGBColor} from "pose-format";
import {PoseRenderer} from "./pose-renderer";
import {h} from "@stencil/core";


export class CanvasPoseRenderer extends PoseRenderer {
  ctx!: CanvasRenderingContext2D;
  thickness!: number;

  renderJoint(_: number, joint: PosePointModel, color: RGBColor) {
    const {R, G, B} = color;
    this.ctx.strokeStyle = `rgba(0, 0, 0, 0)`;
    this.ctx.fillStyle = `rgba(${R}, ${G}, ${B}, ${joint.C})`;

    const radius = this.thickness;
    this.ctx.beginPath();
    this.ctx.arc(this.x(joint.X), this.y(joint.Y), radius, 0, 2 * Math.PI);
    this.ctx.fill();
    this.ctx.stroke();
  }

  renderLimb(from: PosePointModel, to: PosePointModel, color: RGBColor) {
    const {R, G, B} = color;

    this.ctx.lineWidth = this.thickness * 3;
    this.ctx.strokeStyle = `rgba(${R}, ${G}, ${B}, ${(from.C + to.C) / 2})`;

    this.ctx.beginPath();
    this.ctx.moveTo(this.x(from.X), this.y(from.Y));
    this.ctx.lineTo(this.x(to.X), this.y(to.Y));
    this.ctx.stroke();
  }

  // renderLimb(from: PosePointModel, to: PosePointModel, color: RGBColor) {
  //   const {R, G, B} = color;
  //   this.ctx.fillStyle = `rgba(${R}, ${G}, ${B}, ${(from.C + to.C) / 2})`;
  //
  //   const x = this.x((from.X + to.X) / 2);
  //   const y = this.y((from.Y + to.Y) / 2);
  //
  //   const sub = {x: this.x(from.X - to.X), y: this.y(from.Y - to.Y)}
  //
  //   const length = Math.sqrt(Math.pow(sub.x, 2) + Math.pow(sub.y, 2));
  //   const radiusX = Math.floor(length / 2);
  //   const radiusY = this.thickness;
  //   const rotation = Math.floor(Math.atan2(sub.y, sub.x) * 180 / Math.PI);
  //   this.ctx.beginPath();
  //   this.ctx.ellipse(x, y, radiusX, radiusY, rotation, 0, 360);
  //   this.ctx.fill();
  // }

  render(frame: PoseBodyFrameModel) {
    const drawCanvas = () => {
      const canvas = this.viewer.element.shadowRoot.querySelector('canvas');
      if (canvas) {
        // TODO: this is unnecessary, but stencil doesn't apply attributes
        canvas.width = this.viewer.elWidth;
        canvas.height = this.viewer.elHeight;

        this.ctx = canvas.getContext('2d');

        if (this.viewer.background) {
          this.ctx.fillStyle = this.viewer.background;
          this.ctx.fillRect(0, 0, canvas.width, canvas.height);
        } else {
          this.ctx.clearRect(0, 0, canvas.width, canvas.height);
        }

        const w = this.viewer.elWidth - 2 * this.viewer.elPadding.width;
        const h = this.viewer.elHeight - 2 * this.viewer.elPadding.height;
        this.thickness = Math.round(Math.sqrt(w * h) / 300);
        this.renderFrame(frame);
      } else {
        throw new Error("Canvas isn't available before first render")
      }
    };

    try {
      drawCanvas();
    } catch (e) {
      requestAnimationFrame(drawCanvas)
    }


    return (
      <canvas width={this.viewer.elWidth} height={this.viewer.elHeight}/>
    )
  }
}
