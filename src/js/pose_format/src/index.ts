import {PoseBodyModel, PoseHeaderModel} from "./types";
import {Buffer} from "buffer";
import {parsePose} from "./parser";
import * as fs from "fs";

export * from './types';

export class Pose {
  constructor(public header: PoseHeaderModel, public body: PoseBodyModel) {
  }

  static from(buffer: Buffer) {
    const pose = parsePose(buffer);
    return new Pose(pose.header, pose.body);
  }

  static async fromLocal(path: string) {
    const buffer = fs.readFileSync(path);
    return Pose.from(buffer);
  }

  static async fromRemote(url: string, abortController?: AbortController) {
    const init: RequestInit = {};
    if (abortController) {
      init.signal = abortController.signal;
    }
    const res = await fetch(url, init);
    if (!res.ok) {
      let message = res.statusText ?? String(res.status);
      try {
        const json = await res.json();
        message = json.message;
      } catch (e) {
      }
      throw new Error(message);
    }
    const buffer = Buffer.from(await res.arrayBuffer());
    return Pose.from(buffer);
  }
}

