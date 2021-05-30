import {Component, Event, EventEmitter, h, Method, Prop, State} from '@stencil/core';

import {Pose, PoseLimb, PoseModel, PosePointModel, RGBColor} from "pose-utils";


@Component({
  tag: 'pose-viewer',
  styleUrl: 'pose-viewer.css',
  shadow: true
})
export class PoseViewer {
  @Prop() src: string; // Source URL for .pose file

  // MediaElement-like properties
  @Prop({mutable: true}) loop: boolean = false;
  @Prop() autoplay: boolean = true;
  @Prop({mutable: true}) playbackRate: number = 1;

  @State() currentTime: number = NaN; // This affects re-rendering
  duration: number = NaN;
  ended: boolean = false;
  paused: boolean = true;
  readyState: number = 0;

  // MediaElement-like events
  @Event() canplaythrough$: EventEmitter<void>;
  @Event() ended$: EventEmitter<void>;
  @Event() loadeddata$: EventEmitter<void>;
  @Event() loadedmetadata$: EventEmitter<void>;
  @Event() loadstart$: EventEmitter<void>;
  @Event() pause$: EventEmitter<void>;
  @Event() play$: EventEmitter<void>;
  // @Event() ratechange$: EventEmitter<void>;
  // @Event() seeked$: EventEmitter<void>;
  // @Event() seeking$: EventEmitter<void>;
  // @Event() timeupdate$: EventEmitter<void>;


  media: HTMLMediaElement;
  pose: PoseModel;

  private loopInterval: any;

  async componentWillLoad() {
    this.loadstart$.emit();
    this.pose = await Pose.fromRemote(this.src);
    console.log(this.pose);
    // Loaded done events
    this.loadedmetadata$.emit();
    this.loadeddata$.emit();
    this.canplaythrough$.emit();

    this.duration = (this.pose.body.frames.length - 1) / this.pose.body.fps;
    this.currentTime = 0;

    if (this.autoplay) {
      this.play();
    }
  }

  @Method()
  async syncMedia(media: HTMLMediaElement): Promise<void> {
    this.media = media;

    this.media.addEventListener('pause', this.pause.bind(this));
    this.media.addEventListener('play', this.play.bind(this));
    const syncTime = () => this.currentTime = this.frameTime(this.media.currentTime);
    this.media.addEventListener('seek', syncTime);
    this.media.addEventListener('timeupdate', syncTime); // To always keep synced

    // Others
    const updateRate = () => this.playbackRate = this.media.playbackRate;
    this.media.addEventListener('ratechange', updateRate);
    updateRate();

    // Start the pose according to the video
    this.clearInterval();
    if (this.media.paused) {
      this.pause();
    } else {
      this.play();
    }
  }

  frameTime(time: number) {
    return Math.floor(time * this.pose.body.fps) / this.pose.body.fps;
  }

  play() {
    if (!this.paused) {
      this.clearInterval();
    }

    this.paused = false;
    this.play$.emit();

    // Reset clip if exceeded duration
    if (this.currentTime > this.duration) {
      this.currentTime = 0;
    }

    const intervalTime = 1000 / (this.pose.body.fps * this.playbackRate)
    console.log("intervalTime", intervalTime)
    if (this.media) {
      this.loopInterval = setInterval(() => this.currentTime = this.frameTime(this.media.currentTime), intervalTime);
    } else {
      // Add the time passed in an interval.
      let lastTime = Date.now() / 1000;
      this.loopInterval = setInterval(() => {
        const now = Date.now() / 1000;
        this.currentTime += (now - lastTime) * this.playbackRate;
        lastTime = now;
        if (this.currentTime > this.duration) {
          if (this.loop) {
            this.currentTime = this.currentTime % this.duration;
          } else {
            this.ended$.emit();
            this.ended = true;

            this.clearInterval();
          }
        }
      }, intervalTime);
    }
  }

  pause() {
    this.paused = true;
    this.pause$.emit();
    this.clearInterval();
  }

  clearInterval() {
    if (this.loopInterval) {
      clearInterval(this.loopInterval);
    }
  }

  disconnectedCallback() {
    this.clearInterval();
  }

  // Render functions

  isJointValid(joint: PosePointModel) {
    return joint.C > 0;
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
    if (!this.pose || isNaN(this.currentTime)) {
      return "";
    }

    const currentTime = this.currentTime > this.duration ? this.duration : this.currentTime;

    const frameId = Math.floor(currentTime * this.pose.body.fps);
    const frame = this.pose.body.frames[frameId];

    return (
      <svg xmlns="http://www.w3.org/2000/svg" width={this.pose.header.width} height={this.pose.header.height}>
        <g>
          {frame.people.map(person => this.pose.header.components.map(component => {
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


