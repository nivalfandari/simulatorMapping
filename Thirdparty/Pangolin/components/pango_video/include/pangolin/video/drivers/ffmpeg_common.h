#pragma once

#include <pangolin/video/video_exception.h>
#include <pangolin/utils/file_utils.h>

extern "C"
{

// HACK for some versions of FFMPEG
#ifndef INT64_C
#define INT64_C(c) (c ## LL)
#define UINT64_C(c) (c ## ULL)
#endif

#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/pixdesc.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>

// Some versions of FFMPEG define this horrid macro in global scope.
#undef PixelFormat
}

namespace pangolin
{

inline AVPixelFormat FfmpegFmtFromString(const std::string fmt)
{
    const std::string lfmt = ToLowerCopy(fmt);
    if(!lfmt.compare("gray8") || !lfmt.compare("grey8") || !lfmt.compare("grey")) {
        return AV_PIX_FMT_GRAY8;
    }
    return av_get_pix_fmt(lfmt.c_str());
}


#define TEST_PIX_FMT_RETURN(fmt) case AV_PIX_FMT_##fmt: return #fmt;
inline std::string FfmpegFmtToString(const AVPixelFormat fmt)
{
    switch( fmt )
    {
    TEST_PIX_FMT_RETURN(YUV420P);
    TEST_PIX_FMT_RETURN(YUYV422);
    TEST_PIX_FMT_RETURN(RGB24);
    TEST_PIX_FMT_RETURN(BGR24);
    TEST_PIX_FMT_RETURN(YUV422P);
    TEST_PIX_FMT_RETURN(YUV444P);
    TEST_PIX_FMT_RETURN(YUV410P);
    TEST_PIX_FMT_RETURN(YUV411P);
    TEST_PIX_FMT_RETURN(GRAY8);
    TEST_PIX_FMT_RETURN(MONOWHITE);
    TEST_PIX_FMT_RETURN(MONOBLACK);
    TEST_PIX_FMT_RETURN(PAL8);
    TEST_PIX_FMT_RETURN(YUVJ420P);
    TEST_PIX_FMT_RETURN(YUVJ422P);
    TEST_PIX_FMT_RETURN(YUVJ444P);
    TEST_PIX_FMT_RETURN(UYVY422);
    TEST_PIX_FMT_RETURN(UYYVYY411);
    TEST_PIX_FMT_RETURN(BGR8);
    TEST_PIX_FMT_RETURN(BGR4);
    TEST_PIX_FMT_RETURN(BGR4_BYTE);
    TEST_PIX_FMT_RETURN(RGB8);
    TEST_PIX_FMT_RETURN(RGB4);
    TEST_PIX_FMT_RETURN(RGB4_BYTE);
    TEST_PIX_FMT_RETURN(NV12);
    TEST_PIX_FMT_RETURN(NV21);
    TEST_PIX_FMT_RETURN(ARGB);
    TEST_PIX_FMT_RETURN(RGBA);
    TEST_PIX_FMT_RETURN(ABGR);
    TEST_PIX_FMT_RETURN(BGRA);
    TEST_PIX_FMT_RETURN(GRAY16BE);
    TEST_PIX_FMT_RETURN(GRAY16LE);
    TEST_PIX_FMT_RETURN(YUV440P);
    TEST_PIX_FMT_RETURN(YUVJ440P);
    TEST_PIX_FMT_RETURN(YUVA420P);
#ifdef FF_API_VDPAU
    TEST_PIX_FMT_RETURN(VDPAU_H264);
    TEST_PIX_FMT_RETURN(VDPAU_MPEG1);
    TEST_PIX_FMT_RETURN(VDPAU_MPEG2);
    TEST_PIX_FMT_RETURN(VDPAU_WMV3);
    TEST_PIX_FMT_RETURN(VDPAU_VC1);
#endif
    TEST_PIX_FMT_RETURN(RGB48BE );
    TEST_PIX_FMT_RETURN(RGB48LE );
    TEST_PIX_FMT_RETURN(RGB565BE);
    TEST_PIX_FMT_RETURN(RGB565LE);
    TEST_PIX_FMT_RETURN(RGB555BE);
    TEST_PIX_FMT_RETURN(RGB555LE);
    TEST_PIX_FMT_RETURN(BGR565BE);
    TEST_PIX_FMT_RETURN(BGR565LE);
    TEST_PIX_FMT_RETURN(BGR555BE);
    TEST_PIX_FMT_RETURN(BGR555LE);
#if LIBAVFORMAT_VERSION_MAJOR >= 59
    TEST_PIX_FMT_RETURN(VAAPI);
#else
    TEST_PIX_FMT_RETURN(VAAPI_MOCO);
    TEST_PIX_FMT_RETURN(VAAPI_IDCT);
    TEST_PIX_FMT_RETURN(VAAPI_VLD);
#endif
    TEST_PIX_FMT_RETURN(YUV420P16LE);
    TEST_PIX_FMT_RETURN(YUV420P16BE);
    TEST_PIX_FMT_RETURN(YUV422P16LE);
    TEST_PIX_FMT_RETURN(YUV422P16BE);
    TEST_PIX_FMT_RETURN(YUV444P16LE);
    TEST_PIX_FMT_RETURN(YUV444P16BE);
#ifdef FF_API_VDPAU
    TEST_PIX_FMT_RETURN(VDPAU_MPEG4);
#endif
    TEST_PIX_FMT_RETURN(DXVA2_VLD);
    TEST_PIX_FMT_RETURN(RGB444BE);
    TEST_PIX_FMT_RETURN(RGB444LE);
    TEST_PIX_FMT_RETURN(BGR444BE);
    TEST_PIX_FMT_RETURN(BGR444LE);
    TEST_PIX_FMT_RETURN(Y400A   );
    TEST_PIX_FMT_RETURN(NB      );
    default: return "";
    }
}

#undef TEST_PIX_FMT_RETURN

}
