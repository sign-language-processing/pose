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

function getBodyParserV0_1(header: PoseHeaderModel) {
    const _points = header.components.map(c => c.points.length).reduce((a, b) => a + b, 0);
    const _dims = Math.max(...header.components.map(c => c.format.length)) - 1;

    let pointParser: any = newParser()
        .array("points", {
            type: "float",
            length: _dims
        });

    let personParser: any = newParser()
        .array("points", {
            type: pointParser,
            length: _points
        });

    const dataFramesParser = newParser()
        .array("people", {
            type: personParser,
            length: "_people"
        });

    return newParser()
        .seek(header.headerLength)
        .uint16("fps")
        .uint16("_frames")
        .uint16("_people")
        .array("dataFrames", {
            type: dataFramesParser,
            length: "_frames"
        })
}


const headerParser = getHeaderParser();

export function parsePose(buffer: Buffer): PoseModel {
    const header = headerParser.parse(buffer) as unknown as PoseHeaderModel;

    let body: PoseBodyModel;
    switch (header.version) {
        case 0:
            body = getBodyParserV0_0(header).parse(buffer) as unknown as PoseBodyModel;
            break;

        case 0.1:
            body = getBodyParserV0_1(header).parse(buffer) as unknown as PoseBodyModel;;
            break;

        default:
            throw new Error("Parsing this body version is not implemented");
    }

    return {header, body};
}
