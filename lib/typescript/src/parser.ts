import {Parser} from 'binary-parser';
import {PoseBodyModel, PoseHeaderModel, PoseModel} from "./types";


function newParser() {
    return new Parser().endianess("little");
}

function componentHeaderParser() {
    const limbParser = newParser()
        .uint16("from")
        .uint16("to");
    const colorParser = newParser()
        .uint16("R")
        .uint16("G")
        .uint16("B");

    const strParser = newParser()
        .uint16("_chars")
        .string("text", {length: "_chars"});


    return newParser()
        .uint16("_name")
        .string("name", {length: "_name"})
        .uint16("_format",)
        .string("format", {length: "_format"})
        .uint16("_points")
        .uint16("_limbs")
        .uint16("_colors")
        .array("points", {
            type: strParser,
            formatter: (arr: any) => arr.map((item: any) => item.text),
            length: "_points"
        })
        .array("limbs", {
            type: limbParser,
            length: "_limbs"
        })
        .array("colors", {
            type: colorParser,
            length: "_colors"
        });
}

function getHeaderParser() {
    const componentParser = componentHeaderParser();

    return newParser()
        .floatle("version")
        .uint16("width")
        .uint16("height")
        .uint16("depth")
        .uint16("_components")
        .array("components", {
            type: componentParser,
            length: "_components"
        })
        // @ts-ignore
        .saveOffset('headerLength')
}


function getBodyParserV0_0(header: PoseHeaderModel) {
    let personParser: any = newParser()
        .int16("id");
    header.components.forEach(component => {
        let pointParser: any = newParser();
        Array.from(component.format).forEach(c => {
            pointParser = pointParser.floatle(c);
        });

        personParser = personParser.array(component.name, {
            "type": pointParser,
            "length": component._points
        });
    });

    const frameParser = newParser()
        .uint16("_people")
        .array("people", {
            type: personParser,
            length: "_people"
        });

    return newParser()
        .seek(header.headerLength)
        .uint16("fps")
        .uint16("_frames")
        .array("frames", {
            type: frameParser,
            length: "_frames"
        })
}

function parseBodyV0_0(header: PoseHeaderModel, buffer: Buffer): PoseBodyModel {
    return getBodyParserV0_0(header).parse(buffer) as unknown as PoseBodyModel
}

function parseBodyV0_1(header: PoseHeaderModel, buffer: Buffer): PoseBodyModel {
    const _points = header.components.map(c => c.points.length).reduce((a, b) => a + b, 0);
    const _dims = Math.max(...header.components.map(c => c.format.length)) - 1;

    const infoParser = newParser()
        .seek(header.headerLength)
        .uint16("fps")
        .uint16("_frames")
        .uint16("_people");

    const info = infoParser.parse(buffer);

    const dataParser = newParser()
        .seek(header.headerLength + 6)
        .array("data", {
            type: "floatle",
            length: info._frames * info._people * _points * _dims
        })
        // @ts-ignore
        .saveOffset('dataLength');

    const data = dataParser.parse(buffer);

    const confidenceParser = newParser()
        .seek(data.dataLength)
        .array("confidence", {
            type: "floatle",
            length: info._frames * info._people * _points
        });

    const confidence = confidenceParser.parse(buffer);

    const frames: any[] = [];
    for (let i = 0; i < info._frames; i++) {
        const people: any[] = [];
        frames.push({people});
        for (let j = 0; j < info._people; j++) {
            const person: any = {};
            people.push(person);
            let k = 0;
            header.components.forEach(component => {
                person[component.name] = [];

                for (let l = 0; l < component.points.length; l++) {
                    const offset = i * (info._people * _points) + j * _points;
                    person[component.name].push({
                        "X": data.data[offset * 2 + (k + l) * 2],
                        "Y": data.data[offset * 2 + (k + l) * 2 + 1],
                        "C": confidence.confidence[offset + k + l]
                    })
                }
                k += component.points.length;
            });
        }
    }

    return {
        ...info,
        frames
    } as PoseBodyModel;

}

const headerParser = getHeaderParser();

export function parsePose(buffer: Buffer): PoseModel {
    const header = headerParser.parse(buffer) as unknown as PoseHeaderModel;

    let body: PoseBodyModel;
    const version = Math.round(header.version * 1000) / 1000;
    switch (version) {
        case 0:
            body = parseBodyV0_0(header, buffer);
            break;

        case 0.1:
            body = parseBodyV0_1(header, buffer);
            break;

        default:
            throw new Error("Parsing this body version is not implemented - " + header.version);
    }

    return {header, body};
}
