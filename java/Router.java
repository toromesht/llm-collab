// SynapseFlow Router — Java (thread-safe, production)
package synapseflow;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class Router {
    public enum Model { DS_V4, QWEN3_235B, GLM4_PLUS, GROQ_LLAMA, KIMI, GLM4_SJTU, QWEN_SJTU, SJTU_DS_THINK }
    static class Synapse {
        double math=0.5, code=0.5, logic=0.1, knowledge=0.5;
        int lastCorrect, lastWrong, consecutiveWrong;
        boolean banned;
    }
    private final Map<Model,Synapse> w = new ConcurrentHashMap<>();
    private int round;
    public Router() { for (Model m : Model.values()) w.put(m, new Synapse()); }

    private int count(String q, String... ks) {
        return (int)Arrays.stream(ks).filter(q::contains).count();
    }

    public Model route(String q) {
        double math=count(q,"定理","证明","方程","优化","概率");
        double code=count(q,"代码","Python","函数","编程","SQL");
        double logic=count(q,"推理","悖论","说谎","逻辑");
        double know=count(q,"什么是","定义","历史","解释");
        double bestScore=-999; Model best=Model.DS_V4;
        for (var e : w.entrySet()) {
            if (e.getValue().banned) continue;
            Synapse s = e.getValue();
            double score = code*s.code + math*s.math + logic*s.logic + know*s.knowledge;
            if (score > bestScore) { bestScore=score; best=e.getKey(); }
        }
        return best;
    }

    public void update(Model m, String cat, boolean correct) {
        Synapse s = w.get(m);
        double[] wp = cat.equals("math") ? new double[]{s.math} :
                     cat.equals("code") ? new double[]{s.code} :
                     cat.equals("logic") ? new double[]{s.logic} : new double[]{s.knowledge};
        int dt = Math.max(1, round - (correct ? s.lastCorrect : s.lastWrong));
        wp[0] = Math.max(-1, Math.min(1, wp[0] + (correct ? 0.15*Math.exp(-dt/5.0) : -0.10*Math.exp(-dt/3.0))));
        if (wp[0] < -0.3 && ++s.consecutiveWrong >= 5) s.banned = true;
        round++;
    }
}
