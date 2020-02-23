# Header
\[`float` Version]
\[`unsigned short` width]
\[`unsigned short` height]
\[`unsigned short` depth]
\[`unsigned short` Number of Components]

### For every component:
\[`string` Component Name]
\[`char[]` Format]
\[`unsigned short` Number of Points]
\[`unsigned short` Number of Limbs]
\[`unsigned short` Number of Colors]

#### For every point:
\[`string` Point Name]

#### For every limb:
\[`unsigned short` From Point Index]
\[`unsigned short` To Point Index]

#### For every color:
\[`unsigned short` Red]
\[`unsigned short` Green]
\[`unsigned short` Blue]

# Body
\[`unsined short` FPS]
\[`unsined short` Number of frames]

## For every frame
\[`unsigned short` Number of People]

#### For every person:
\[`short` Person ID]

##### For every person's component:
\[`float` X]
\[`float` Y]
\[`float` Confidence]
