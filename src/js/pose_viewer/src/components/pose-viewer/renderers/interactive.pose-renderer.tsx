import { PoseRenderer } from "./pose-renderer";
import { PoseBodyFrameModel, PosePointModel, RGBColor } from "pose-format";
import { h } from "@stencil/core";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";

export class InteractivePoseRenderer extends PoseRenderer {
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private poseGroup!: THREE.Group;
  // Reuse the same geometry for all joints.
  private jointGeometry = new THREE.SphereGeometry(2, 16, 16);

  // Convert a joint to a 3D point.
  private transform(joint: PosePointModel): THREE.Vector3 {
    const { width, height } = this.viewer.pose.header;
    const x = joint.X - width / 2;
    const y = height / 2 - joint.Y;
    const z = joint.Z;
    return new THREE.Vector3(x, y, z);
  }

  // Create a mesh for a joint.
  public renderJoint(_: number, joint: PosePointModel, color: RGBColor): THREE.Mesh {
    const material = new THREE.MeshBasicMaterial({
      color: new THREE.Color(color.R / 255, color.G / 255, color.B / 255),
      transparent: true,
      opacity: joint.C,
    });
    const sphere = new THREE.Mesh(this.jointGeometry, material);
    sphere.position.copy(this.transform(joint));
    return sphere;
  }

  // Create a line between two joints.
  public renderLimb(from: PosePointModel, to: PosePointModel, color: RGBColor): THREE.Line {
    const material = new THREE.LineBasicMaterial({
      color: new THREE.Color(color.R / 255, color.G / 255, color.B / 255),
      transparent: true,
      opacity: (from.C + to.C) / 2,
    });
    const points = [this.transform(from), this.transform(to)];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    return new THREE.Line(geometry, material);
  }

  // Initialize Three.js components.
  private initThree(canvas: HTMLCanvasElement): void {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setSize(this.viewer.elWidth, this.viewer.elHeight);

    this.scene = new THREE.Scene();
    if (this.viewer.background) {
      this.scene.background = new THREE.Color(this.viewer.background);
    }

    this.camera = new THREE.PerspectiveCamera(
      75,
      this.viewer.elWidth / this.viewer.elHeight,
      0.1,
      1000
    );
    const size = Math.max(this.viewer.pose.header.width, this.viewer.pose.header.height);
    this.camera.position.set(0, 0, size);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    this.scene.add(ambientLight);

    this.poseGroup = new THREE.Group();
    this.scene.add(this.poseGroup);

    this.animate();
  }

  // Continuous render loop.
  private animate = (): void => {
    requestAnimationFrame(this.animate);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  // Helper to dispose objects and free resources.
  private disposeObject(obj: THREE.Object3D): void {
    if (obj instanceof THREE.Mesh) {
      // Avoid disposing shared geometry.
      if (obj.geometry !== this.jointGeometry) {
        obj.geometry.dispose();
      }
      if (Array.isArray(obj.material)) {
        obj.material.forEach(mat => mat.dispose());
      } else {
        obj.material.dispose();
      }
    } else if (obj instanceof THREE.Line) {
      obj.geometry.dispose();
      obj.material.dispose();
    }
  }

  // Update the scene with the latest pose frame.
  private updateScene(frame: PoseBodyFrameModel): void {
    // Dispose of previous objects to avoid memory leaks.
    this.poseGroup.children.forEach(child => this.disposeObject(child));
    this.poseGroup.clear();

    const rendered = this.renderFrame(frame);
    const objects: THREE.Object3D[] = rendered.flat(Infinity).filter(
      (o): o is THREE.Object3D => o instanceof THREE.Object3D
    );
    objects.forEach(obj => this.poseGroup.add(obj));

    // Update orbit controls target.
    this.poseGroup.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(this.poseGroup);
    const center = new THREE.Vector3();
    box.getCenter(center);
    this.controls.target.copy(center);
  }

  // Main render method.
  public render(frame: PoseBodyFrameModel) {
    const initRenderer = (): void => {
      const canvas = this.viewer.element.shadowRoot.querySelector("canvas") as HTMLCanvasElement;
      if (canvas && !this.renderer) {
        this.initThree(canvas);
      }
      if (this.renderer) {
        this.updateScene(frame);
      } else {
        requestAnimationFrame(initRenderer);
      }
    };
    initRenderer();
    return (
      <canvas width={this.viewer.elWidth} height={this.viewer.elHeight} />
    );
  }
}
