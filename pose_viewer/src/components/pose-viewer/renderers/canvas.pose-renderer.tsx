import {PoseBodyFrameModel, PosePointModel, RGBColor} from "pose-format";
import {PoseRenderer} from "./pose-renderer";
import {h} from "@stencil/core";


export class CanvasPoseRenderer extends PoseRenderer {
  ctx!: CanvasRenderingContext2D;

  renderJoint(_: number, joint: PosePointModel, color: RGBColor) {
    const {R, G, B} = color;
    this.ctx.strokeStyle = `rgba(0, 0, 0, ${joint.C})`;
    this.ctx.fillStyle = `rgba(${R}, ${G}, ${B}, ${joint.C})`;
    this.ctx.lineWidth = 1;

    this.ctx.beginPath();
    this.ctx.arc(this.x(joint.X), this.y(joint.Y), 4, 0, 2 * Math.PI);
    this.ctx.fill();
    this.ctx.stroke();
  }

  renderLimb(from: PosePointModel, to: PosePointModel, color: RGBColor) {
    const {R, G, B} = color;
    this.ctx.strokeStyle = `rgba(${R}, ${G}, ${B}, ${(from.C + to.C) / 2})`;
    this.ctx.lineWidth = 8;

    this.ctx.beginPath();
    this.ctx.moveTo(this.x(from.X), this.y(from.Y));
    this.ctx.lineTo(this.x(to.X), this.y(to.Y));
    this.ctx.stroke();
  }

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
