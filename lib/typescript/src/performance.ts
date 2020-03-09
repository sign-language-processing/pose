import * as fs from 'fs';
import {Pose} from "./index";

const testJSONFile = fs.readFileSync("../../../sample-data/json/video_000000000000_keypoints.json")
const testPoseFile = fs.readFileSync("../../../sample-data/imgs/video_000000000000_keypoints.pose")


console.time("pose")
for (let i = 0; i < 100; i++) {
    Pose.from(testPoseFile)
}
console.timeEnd("pose")

console.time("json")
for (let i = 0; i < 100; i++) {
    JSON.parse(String(testJSONFile));
}
console.timeEnd("json")
